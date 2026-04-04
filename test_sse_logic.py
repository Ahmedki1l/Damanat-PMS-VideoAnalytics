
import asyncio
import json
from src.events.event_bus import EventBus
from src.models.state_machine import SlotEvent
from fastapi.sse import EventSourceResponse

async def test_sse_logic():
    print("--- Testing EventBus + SSE Logic ---")
    
    # 1. Setup EventBus
    bus = EventBus()
    
    # 2. Setup Async Queue and Subscriber (similar to api.py)
    queue = asyncio.Queue()
    def callback(event):
        # We use call_soon_threadsafe to push to the async queue from the engine thread
        # For this test, we skip the loop check and put directly
        queue.put_nowait(event)
        
    bus.subscribe(callback)
    print("[OK] Subscribed to EventBus")
    
    # 3. Emit a test event
    from datetime import datetime
    test_event = SlotEvent(
        event_type="vehicle_violation",
        slot_id="A1",
        track_id=123,
        timestamp=datetime.now().isoformat(),
        camera_id="CAM_01",
        floor="B1"
    )
    bus.emit(test_event)
    print("[OK] Emitted violation event")
    
    # 4. Check if Queue received it
    received = await queue.get()
    assert received.slot_id == "A1"
    assert received.event_type == "vehicle_violation"
    print("[OK] Queue received the correct event")
    
    # 5. Verify EventSourceResponse formatting
    # We mimic the generator from api.py
    async def event_generator():
        yield {"data": received.to_dict()}
    
    response = EventSourceResponse(event_generator())
    
    # Iterate through the response to see the raw output
    print("\n--- Raw SSE Stream Output ---")
    async for line in response.body_iterator:
        print(f"Line: {repr(line)}")
        if b"data:" in line:
            # Verify it's valid JSON
            # The output of EventSourceResponse will be something like: data: {"slot_id": "A1", ...}\n\n
            line_str = line.decode()
            if line_str.startswith("data: "):
                data_str = line_str.replace("data: ", "").strip()
                data_json = json.loads(data_str)
                assert data_json["slot_id"] == "A1"
                print(f"[OK] EventSourceResponse correctly formatted the JSON: {data_json}")
    
    print("\n--- ALL SSE LOGIC TESTS PASSED ---")

if __name__ == "__main__":
    asyncio.run(test_sse_logic())
