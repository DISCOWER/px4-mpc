#!/usr/bin/env python
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.telemetry import FlightMode


CONNECTION_ADDRESSES = [
    "udpin://0.0.0.0:14540",
    "udpin://0.0.0.0:14550",
]

CONNECTION_TIMEOUT = 2  # seconds


async def try_connect(address: str, timeout: float):
    drone = System()
    try:
        await asyncio.wait_for(drone.connect(system_address=address), timeout)
        print(f"Connected successfully to {address}")
        return drone
    except asyncio.TimeoutError:
        print(f"Connection to {address} timed out.")
    except Exception as e:
        print(f"Connection to {address} failed with error: {e}")
    return None


async def connect_to_any():
    for address in CONNECTION_ADDRESSES:
        print(f"Trying to connect to {address}...")
        drone = await try_connect(address, CONNECTION_TIMEOUT)
        if drone:
            return drone
    print("Failed to connect to any address.")
    return None


async def run():

    print("-- Starting PX4 MPC Vehicle Manager. Finding vehicle...")
    drone = await connect_to_any()
    if not drone:
        print("No vehicle found. Exiting.")
        return

    status_text_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for vehicle to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to vehicle!")
            break

    # Start Offboard mode
    print("-- Starting offboard")
    async for mode in drone.telemetry.flight_mode():
        print(f"Flight mode: {mode}")
        if mode == FlightMode.OFFBOARD:
            print("Vehicle is in OFFBOARD mode.")
            break
        else:
            print("Vehicle is not in OFFBOARD mode. Starting offboard mode...")
            try:
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                )
                await drone.offboard.start()
                print("-- Offboard mode started successfully")
                break
            except OffboardError as error:
                print(
                    f"Starting offboard mode failed with error code: \
                    {error._result.result}"
                )
                print("Retrying in 5 seconds...")
                await asyncio.sleep(5)

    vehicle_armed = False
    print("-- Waiting for vehicle to arm")
    async for is_armed in drone.telemetry.armed():
        print(f"Vehicle armed: {is_armed}")

        if is_armed:
            print("Vehicle is armed!")
            vehicle_armed = True
            break
        else:
            print("Arming vehicle...")
            try:
                await drone.action.arm()
            except Exception as e:
                print(f"Failed to arm vehicle: {e}")
                print("Retrying in 5 seconds...")
                await asyncio.sleep(5)


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


def main():
    # Run the asyncio loop
    asyncio.run(run())


if __name__ == "__main__":
    main()
