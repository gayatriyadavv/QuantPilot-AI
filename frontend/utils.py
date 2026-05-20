import asyncio

def run_async(coro):
    """Safely run async coroutines in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # Apply nest_asyncio if needed, or create a new thread
        import nest_asyncio
        nest_asyncio.apply()
        
    return loop.run_until_complete(coro)
