import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from dateutil.parser import parse as dateparse

from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.candles_feed.btc_markets_spot_candles import constants as CONSTANTS
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.logger import HummingbotLogger


class BtcMarketsSpotCandles(CandlesBase):
    """
    BTC Markets implementation for fetching candlestick data.

    Note: BTC Markets doesn't support WebSocket for candles, so we use polling instead.
    This implementation creates "heartbeat" candles at regular intervals that get replaced
    with actual data when available, maintaining the equidistant requirement of CandlesBase.
    """

    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self, trading_pair: str, interval: str = "1m", max_records: int = 150):
        super().__init__(trading_pair, interval, max_records)
        self._last_received_timestamp = None
        self._consecutive_empty_responses = 0
        self._fast_polling_mode = False
        self._expected_next_candle_time = None
        self._historical_fill_in_progress = False
        self._last_real_candle = None  # Store the last candle with actual trading data

    @property
    def name(self):
        return f"btc_markets_{self._trading_pair}"

    @property
    def rest_url(self):
        return CONSTANTS.REST_URL

    @property
    def wss_url(self):
        # BTC Markets doesn't support WebSocket for candles
        return CONSTANTS.WSS_URL

    @property
    def health_check_url(self):
        return self.rest_url + CONSTANTS.HEALTH_CHECK_ENDPOINT

    @property
    def candles_url(self):
        market_id = self.get_exchange_trading_pair(self._trading_pair)
        return self.rest_url + CONSTANTS.CANDLES_ENDPOINT.format(market_id=market_id)

    @property
    def candles_endpoint(self):
        return CONSTANTS.CANDLES_ENDPOINT

    @property
    def candles_max_result_per_rest_request(self):
        return CONSTANTS.MAX_RESULTS_PER_CANDLESTICK_REST_REQUEST

    @property
    def rate_limits(self):
        return CONSTANTS.RATE_LIMITS

    @property
    def intervals(self):
        return CONSTANTS.INTERVALS

    async def check_network(self) -> NetworkStatus:
        rest_assistant = await self._api_factory.get_rest_assistant()
        await rest_assistant.execute_request(
            url=self.health_check_url, throttler_limit_id=CONSTANTS.HEALTH_CHECK_ENDPOINT
        )
        return NetworkStatus.CONNECTED

    def get_exchange_trading_pair(self, trading_pair):
        """
        Converts from the Hummingbot trading pair format to the exchange's trading pair format.
        BTC Markets uses the same format so no conversion is needed.
        """
        return trading_pair

    @property
    def _is_first_candle_not_included_in_rest_request(self):
        return False

    @property
    def _is_last_candle_not_included_in_rest_request(self):
        return False

    def _get_rest_candles_params(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None
    ) -> dict:
        """
        Generates parameters for the REST API request to fetch candles.

        BTC Markets API documentation:
        - timeWindow: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 3h, 4h, 6h, 1d, 1w, 1mo
        - from/to: ISO 8601 timestamps (e.g., 2018-08-20T06:22:11.000000Z)
        - Can use either pagination (before/after/limit) or timestamps (from/to)
        - Maximum 1000 items when using timestamps
        """
        params = {
            "timeWindow": self.intervals[self.interval],
        }

        # If we're not specifying time parameters, use pagination with a small limit
        if start_time is None and end_time is None:
            params["limit"] = limit if limit is not None else 10  # Start with small limit for testing
        else:
            # Use timestamp parameters
            params["limit"] = min(limit if limit is not None else 1000, 1000)  # Max 1000

            if start_time:
                start_iso = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                params["from"] = start_iso

            if end_time:
                end_iso = datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                params["to"] = end_iso

        self.logger().debug(f"Candles request params: {params}")
        return params

    def _parse_rest_candles(self, data: List[List[str]], end_time: Optional[int] = None) -> List[List[float]]:
        """
        Parse the REST API response into the standard candle format.

        BTC Markets response format:
        [
            [
                "2019-09-02T18:00:00.000000Z",  # timestamp
                "15100",                         # open
                "15200",                         # high
                "15100",                         # low
                "15199",                         # close
                "4.11970335"                     # volume
            ],
            ...
        ]

        Convert to standard format:
        [timestamp, open, high, low, close, volume, quote_asset_volume, n_trades, taker_buy_base_volume, taker_buy_quote_volume]
        """
        if not isinstance(data, list):
            self.logger().warning(f"Expected list, got {type(data)}: {data}")
            return []

        if len(data) == 0:
            self.logger().debug("Received empty candles data from API")
            return []

        self.logger().debug(f"Parsing {len(data)} candles from REST API response")

        new_hb_candles = []
        for i, candle in enumerate(data):
            try:
                if not isinstance(candle, list) or len(candle) < 6:
                    self.logger().warning(f"Invalid candle format at index {i}: {candle}")
                    continue

                # Parse timestamp from ISO 8601 format
                timestamp = self.ensure_timestamp_in_seconds(dateparse(candle[0]).timestamp())
                open_price = float(candle[1])
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                volume = float(candle[5])

                # BTC Markets doesn't provide these values, so we set them to 0
                quote_asset_volume = 0.0
                n_trades = 0.0
                taker_buy_base_volume = 0.0
                taker_buy_quote_volume = 0.0

                new_hb_candles.append(
                    [
                        timestamp,
                        open_price,
                        high,
                        low,
                        close,
                        volume,
                        quote_asset_volume,
                        n_trades,
                        taker_buy_base_volume,
                        taker_buy_quote_volume,
                    ]
                )

            except Exception as e:
                self.logger().error(f"Error parsing candle {candle}: {e}")

        # Sort by timestamp in ascending order (oldest first, newest last)
        new_hb_candles.sort(key=lambda x: x[0])

        self.logger().debug(f"Parsed {len(new_hb_candles)} candles successfully")
        return new_hb_candles

    def _create_heartbeat_candle(self, timestamp: float, reference_candle: Optional[List[float]] = None) -> List[float]:
        """
        Create a "heartbeat" candle for periods with no trading activity.

        Uses the close price from the reference candle (or last real candle) as OHLC
        with zero volume to indicate no trading activity.

        Args:
            timestamp: The timestamp for the heartbeat candle
            reference_candle: Reference candle to use for price data (optional)

        Returns:
            A candle array in the standard format
        """
        if reference_candle is not None:
            # Use the close price from the reference candle
            close_price = reference_candle[4]  # close price
        elif self._last_real_candle is not None:
            # Use the close price from the last real candle
            close_price = self._last_real_candle[4]
        elif len(self._candles) > 0:
            # Use the close price from the last candle in our deque
            close_price = self._candles[-1][4]
        else:
            # Fallback: use a placeholder price (should rarely happen)
            close_price = 0.0

        # Create heartbeat candle: OHLC all the same (no price movement), zero volume
        heartbeat = [
            timestamp,  # timestamp
            close_price,  # open = last close
            close_price,  # high = last close
            close_price,  # low = last close
            close_price,  # close = last close
            0.0,  # volume = 0 (no trades)
            0.0,  # quote_asset_volume = 0
            0.0,  # n_trades = 0
            0.0,  # taker_buy_base_volume = 0
            0.0,  # taker_buy_quote_volume = 0
        ]

        # Debug logging
        dt = datetime.fromtimestamp(timestamp)
        self.logger().debug(f"Created heartbeat candle: {timestamp} ({dt}) price: {close_price}")

        return heartbeat

    def _fill_gaps_with_heartbeats(
        self, candles: List[List[float]], start_timestamp: float, end_timestamp: float
    ) -> List[List[float]]:
        """
        Fill gaps in candle data with heartbeat candles to maintain equidistant intervals.

        Args:
            candles: List of actual candles from the API
            start_timestamp: Start of the period we want to fill
            end_timestamp: End of the period we want to fill

        Returns:
            Complete list of candles with heartbeats filling any gaps
        """
        self.logger().debug(f"Filling gaps from {start_timestamp} to {end_timestamp}")
        self.logger().debug(f"Input: {len(candles)} real candles")

        if len(candles) == 0:
            # No real candles, create all heartbeats
            result = []
            current_timestamp = self._round_timestamp_to_interval_multiple(start_timestamp)
            interval_count = 0
            while current_timestamp <= end_timestamp:
                heartbeat = self._create_heartbeat_candle(current_timestamp)
                result.append(heartbeat)
                current_timestamp += self.interval_in_seconds
                interval_count += 1
                if interval_count > 1000:  # Safety check
                    self.logger().warning("Too many intervals generated, breaking")
                    break

            self.logger().debug(f"Generated {len(result)} heartbeat candles")
            return result

        # Create a map of real candles by timestamp
        candle_map = {}
        for c in candles:
            rounded_ts = self._round_timestamp_to_interval_multiple(c[0])
            candle_map[rounded_ts] = c

        self.logger().debug(f"Real candle timestamps: {list(candle_map.keys())}")

        # Fill the complete time range
        result = []
        current_timestamp = self._round_timestamp_to_interval_multiple(start_timestamp)
        interval_count = 0

        while current_timestamp <= end_timestamp:
            if current_timestamp in candle_map:
                # We have real data for this timestamp
                real_candle = candle_map[current_timestamp]
                result.append(real_candle)
                self._last_real_candle = real_candle  # Update last real candle reference
                dt = datetime.fromtimestamp(current_timestamp)
                self.logger().debug(f"Added real candle: {current_timestamp} ({dt}) vol: {real_candle[5]}")
            else:
                # Create heartbeat candle for this timestamp
                heartbeat = self._create_heartbeat_candle(current_timestamp)
                result.append(heartbeat)

            current_timestamp += self.interval_in_seconds
            interval_count += 1
            if interval_count > 1000:  # Safety check
                self.logger().warning("Too many intervals generated, breaking")
                break

        self.logger().debug(
            f"Generated {len(result)} total candles ({len(candles)} real, {len(result) - len(candles)} heartbeats)"
        )

        # Verify timestamps are correct
        if len(result) > 1:
            first_ts = result[0][0]
            last_ts = result[-1][0]
            expected_count = int((last_ts - first_ts) / self.interval_in_seconds) + 1
            self.logger().debug(
                f"Timestamp verification: first={first_ts}, last={last_ts}, count={len(result)}, expected={expected_count}"
            )

        return result

    async def fill_historical_candles(self):
        """
        Fill historical candles with heartbeats to maintain equidistant intervals.
        This creates a complete time series that satisfies CandlesBase validation.
        """
        if self._historical_fill_in_progress:
            return

        self._historical_fill_in_progress = True

        try:
            iteration = 0
            max_iterations = 20

            while not self.ready and len(self._candles) > 0 and iteration < max_iterations:
                iteration += 1

                try:
                    oldest_timestamp = self._candles[0][0]
                    missing_records = self._candles.maxlen - len(self._candles)

                    if missing_records <= 0:
                        break

                    # Calculate the time range we need to fill
                    end_timestamp = oldest_timestamp - self.interval_in_seconds  # One interval before oldest
                    start_timestamp = end_timestamp - (missing_records * self.interval_in_seconds)

                    self.logger().debug(f"=== HISTORICAL FILL ITERATION {iteration} ===")
                    self.logger().debug(
                        f"Current oldest timestamp: {oldest_timestamp} ({datetime.fromtimestamp(oldest_timestamp)})"
                    )
                    self.logger().debug(f"Missing records: {missing_records}")
                    self.logger().debug(f"Filling range: {start_timestamp} to {end_timestamp}")
                    self.logger().debug(f"  Start: {datetime.fromtimestamp(start_timestamp)}")
                    self.logger().debug(f"  End: {datetime.fromtimestamp(end_timestamp)}")

                    # Fetch real candles for this time range
                    real_candles = await self.fetch_candles(
                        start_time=start_timestamp, end_time=end_timestamp + self.interval_in_seconds
                    )

                    self.logger().debug(f"Fetched {len(real_candles)} real candles from API")
                    if len(real_candles) > 0:
                        real_timestamps = [c[0] for c in real_candles]
                        self.logger().debug(f"Real candle timestamps: {real_timestamps}")

                    # Fill gaps with heartbeats
                    complete_candles = self._fill_gaps_with_heartbeats(real_candles, start_timestamp, end_timestamp)

                    # Add the complete candles to our deque
                    if len(complete_candles) > 0:
                        # Take only what we need
                        candles_to_add = (
                            complete_candles[-missing_records:]
                            if len(complete_candles) > missing_records
                            else complete_candles
                        )

                        self.logger().debug(f"Adding {len(candles_to_add)} candles to deque")

                        # Debug the timestamps we're about to add
                        if len(candles_to_add) > 0:
                            add_timestamps = [c[0] for c in candles_to_add]
                            self.logger().debug(f"Timestamps to add: {add_timestamps[:5]}...")  # Show first 5

                        # Add them in reverse order to maintain chronological order
                        for i, candle in enumerate(reversed(candles_to_add)):
                            self._candles.appendleft(candle)
                            if i < 3:  # Debug first few
                                self.logger().debug(
                                    f"  Added candle {i}: {candle[0]} ({datetime.fromtimestamp(candle[0])})"
                                )

                        self.logger().debug(f"Total candles after adding: {len(self._candles)}")

                        # Check the new deque state
                        if len(self._candles) >= 3:
                            first_three = [self._candles[i][0] for i in range(3)]
                            self.logger().debug(f"First 3 timestamps in deque: {first_three}")
                    else:
                        # No more data available
                        self.logger().debug("No candles generated, breaking")
                        break

                    await self._sleep(0.1)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger().exception(f"Error during historical fill iteration {iteration}: {e}")
                    await self._sleep(1.0)

            self.logger().debug(f"Historical fill completed with {len(self._candles)} candles")

        finally:
            self._historical_fill_in_progress = False

    async def listen_for_subscriptions(self):
        """
        Since BTC Markets doesn't support WebSocket for candles, we implement polling
        with heartbeat generation to maintain equidistant intervals.
        """
        self.logger().info(f"Starting polling for {self._trading_pair} candles (WebSocket not supported)")

        # Initial setup
        await self._initialize_candles()

        while True:
            try:
                poll_interval = self._get_poll_interval()
                await self._poll_and_update_candles()
                await self._sleep(poll_interval)

            except asyncio.CancelledError:
                self.logger().info("Polling cancelled")
                raise
            except Exception as e:
                self.logger().exception(f"Unexpected error during polling: {e}")
                await self._sleep(10.0)

    def _get_poll_interval(self) -> float:
        """Calculate appropriate polling interval."""
        current_time = self._time()

        if len(self._candles) == 0:
            return 2.0

        if self._expected_next_candle_time:
            time_until_next_candle = self._expected_next_candle_time - current_time

            if time_until_next_candle <= 30:
                return 2.0

            return min(30, self.interval_in_seconds / 2)

        return min(30, self.interval_in_seconds)

    async def _poll_and_update_candles(self):
        """
        Poll for latest candles and update our data, creating heartbeats as needed.
        This method ensures we always advance time with proper heartbeats.
        """
        try:
            current_time = self._time()

            # First, ensure we have heartbeats up to the current time
            await self._ensure_heartbeats_to_current_time(current_time)

            # Then fetch and process any real data
            recent_candles = await self.fetch_recent_candles(limit=5)

            if len(recent_candles) == 0:
                self._consecutive_empty_responses += 1
                return

            self._consecutive_empty_responses = 0

            if len(self._candles) == 0:
                # First candle - initialize
                latest_candle = recent_candles[-1]
                self._candles.append(latest_candle)
                self._last_real_candle = latest_candle
                self._update_expected_next_candle_time(latest_candle[0])
                self.logger().debug(f"Initialized with first candle: {latest_candle[0]}")
                safe_ensure_future(self.fill_historical_candles())
                return

            # Process any new real candles and replace heartbeats where appropriate
            await self._process_new_real_candles(recent_candles)

        except Exception as e:
            self.logger().error(f"Error during polling: {e}")
            self._consecutive_empty_responses += 1

    async def _ensure_heartbeats_to_current_time(self, current_time: float):
        """
        Ensure we have heartbeats (or real candles) up to the current time interval.
        This maintains the continuous time progression.
        """
        if len(self._candles) == 0:
            return

        # Calculate what the current interval timestamp should be
        current_interval_timestamp = self._round_timestamp_to_interval_multiple(current_time)
        last_candle_timestamp = self._candles[-1][0]

        # Create heartbeats for any missing intervals up to current time
        next_expected_timestamp = last_candle_timestamp + self.interval_in_seconds

        while next_expected_timestamp <= current_interval_timestamp:
            # Check if we already have a candle at this timestamp
            existing_candle = next((c for c in self._candles if c[0] == next_expected_timestamp), None)

            if existing_candle is None:
                # Create heartbeat for this timestamp
                heartbeat = self._create_heartbeat_candle(next_expected_timestamp)
                self._candles.append(heartbeat)
                self.logger().debug(f"Added heartbeat for current time progression: {next_expected_timestamp}")

            next_expected_timestamp += self.interval_in_seconds

    async def _fill_gap_to_timestamp(self, target_timestamp: int):
        """
        Fill any gap between the current latest candle and the target timestamp with heartbeats.

        Args:
            target_timestamp: The timestamp we want to ensure we have candles up to
        """
        if len(self._candles) == 0:
            return

        current_latest_timestamp = int(self._candles[-1][0])

        # If target is not ahead of current, no gap to fill
        if target_timestamp <= current_latest_timestamp:
            return

        # Calculate the next expected timestamp after our current latest
        next_expected_timestamp = current_latest_timestamp + self.interval_in_seconds

        # Fill heartbeats for each missing interval up to (but not including) the target
        while next_expected_timestamp < target_timestamp:
            heartbeat = self._create_heartbeat_candle(next_expected_timestamp)
            self._candles.append(heartbeat)
            self.logger().debug(f"Filled gap with heartbeat at {next_expected_timestamp}")
            next_expected_timestamp += self.interval_in_seconds

    async def _process_new_real_candles(self, recent_candles: List[List[float]]):
        """
        Process new real candles from the API and replace heartbeats where appropriate.
        """
        current_latest_timestamp = int(self._candles[-1][0])

        # Look for candles newer than our current latest
        new_candles = [c for c in recent_candles if int(c[0]) > current_latest_timestamp]

        if len(new_candles) > 0:
            # We have genuinely new real data
            for new_candle in new_candles:
                new_timestamp = int(new_candle[0])

                # First ensure we have heartbeats up to this new timestamp
                await self._fill_gap_to_timestamp(new_timestamp)

                # Now replace the heartbeat at this timestamp with real data (or add if missing)
                self._replace_or_add_candle_at_timestamp(new_timestamp, new_candle)

                self._last_real_candle = new_candle
                self._update_expected_next_candle_time(new_candle[0])
                self.logger().debug(f"Added/updated real candle: {new_candle[0]}")
        else:
            # No new candles, but check if we need to update an existing one
            current_candle = next((c for c in recent_candles if int(c[0]) == current_latest_timestamp), None)
            if current_candle is not None:
                # Check if this real candle has changed (price/volume updates)
                last_candle_index = len(self._candles) - 1
                if not np.array_equal(self._candles[last_candle_index], current_candle):
                    self._candles[last_candle_index] = current_candle
                    self._last_real_candle = current_candle
                    self.logger().debug(f"Updated existing candle: {current_candle[0]}")

    def _replace_or_add_candle_at_timestamp(self, timestamp: int, real_candle: List[float]):
        """
        Replace a heartbeat with real data, or add a new candle if it doesn't exist.
        """
        # Find if we already have a candle at this timestamp
        for i, existing_candle in enumerate(self._candles):
            if int(existing_candle[0]) == timestamp:
                # Replace the existing candle (likely a heartbeat) with real data
                self._candles[i] = real_candle
                self.logger().debug(f"Replaced heartbeat with real data at {timestamp}")
                return

        # If we don't have a candle at this timestamp, add it in the correct position
        # Find the correct insertion point to maintain chronological order
        insert_index = len(self._candles)
        for i, existing_candle in enumerate(self._candles):
            if existing_candle[0] > timestamp:
                insert_index = i
                break

        # Insert the new candle at the correct position
        self._candles.insert(insert_index, real_candle)
        self.logger().debug(f"Inserted new real candle at position {insert_index}, timestamp {timestamp}")

    async def _maybe_create_heartbeat(self):
        """
        This method is now handled by _ensure_heartbeats_to_current_time.
        Kept for compatibility but functionality moved.
        """
        # This functionality is now handled in _ensure_heartbeats_to_current_time
        pass

    async def fetch_recent_candles(self, limit: int = 10) -> List[List[float]]:
        """Fetch recent candles using pagination."""
        try:
            params = {"timeWindow": self.intervals[self.interval], "limit": limit}

            rest_assistant = await self._api_factory.get_rest_assistant()
            response = await rest_assistant.execute_request(
                url=self.candles_url,
                throttler_limit_id=self._rest_throttler_limit_id,
                params=params,
                method=self._rest_method,
            )

            return self._parse_rest_candles(response)

        except Exception as e:
            self.logger().error(f"Error fetching recent candles: {e}")
            return []

    async def _initialize_candles(self):
        """Initialize with recent candle data."""
        try:
            self.logger().info("Initializing candles with recent data...")

            candles = await self.fetch_recent_candles(limit=5)

            if len(candles) > 0:
                latest_candle = candles[-1]
                self._candles.append(latest_candle)
                self._last_real_candle = latest_candle
                self._ws_candle_available.set()
                self._update_expected_next_candle_time(latest_candle[0])

                # Debug: Print actual timestamp values
                timestamp_formatted = datetime.fromtimestamp(latest_candle[0])
                self.logger().info(f"Initialized with candle at timestamp: {latest_candle[0]} ({timestamp_formatted})")

                safe_ensure_future(self.fill_historical_candles())
            else:
                self.logger().warning("No recent candles found during initialization")

        except Exception as e:
            self.logger().error(f"Failed to initialize candles: {e}")

    def _update_expected_next_candle_time(self, current_candle_timestamp: float):
        """Calculate when we expect the next candle."""
        self._expected_next_candle_time = current_candle_timestamp + self.interval_in_seconds

    def ws_subscription_payload(self):
        """Not used for BTC Markets since WebSocket is not supported for candles."""
        raise NotImplementedError("WebSocket not supported for BTC Markets candles")

    def _parse_websocket_message(self, data):
        """Not used for BTC Markets since WebSocket is not supported for candles."""
        raise NotImplementedError("WebSocket not supported for BTC Markets candles")
