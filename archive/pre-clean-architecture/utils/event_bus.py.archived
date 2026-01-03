"""
Event Bus for Inter-Agent Communication
Async event-driven messaging system for agent coordination
"""

import asyncio
from typing import Dict, List, Callable, Optional, Any, Set, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import fnmatch
import logging
from collections import defaultdict


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class Event:
    """Event data structure"""
    type: str
    data: Dict[str, Any]
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class EventBus:
    """
    Asynchronous event bus for agent communication
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.async_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Event filtering
        self.filters: Dict[str, Callable] = {}
        
        # Event history for debugging
        self.event_history: List[Event] = []
        self.max_history_size = 100
        
        # Metrics
        self.metrics = {
            'events_published': 0,
            'events_delivered': 0,
            'events_dropped': 0,
            'active_subscriptions': 0
        }
        
        # Pattern subscriptions (for wildcard matching)
        self.pattern_subscribers: List[tuple[str, Callable]] = []
        
        # Event processing task
        self.processor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the event bus processor"""
        if not self.running:
            self.running = True
            self.processor_task = asyncio.create_task(self._process_events())
            self.logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus gracefully"""
        self.running = False
        
        if self.processor_task:
            # Wait for queue to empty
            await self.event_queue.join()
            self.processor_task.cancel()
            
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Event bus stopped")
    
    def subscribe(self, event_type: str, handler: Callable,
                  filter_func: Optional[Callable] = None):
        """
        Subscribe to an event type
        
        Args:
            event_type: Event type or pattern (supports wildcards)
            handler: Callback function to handle events
            filter_func: Optional filter function for events
        """
        if '*' in event_type or '?' in event_type:
            # Pattern subscription
            self.pattern_subscribers.append((event_type, handler))
        else:
            # Direct subscription
            if asyncio.iscoroutinefunction(handler):
                self.async_subscribers[event_type].append(handler)
            else:
                self.subscribers[event_type].append(handler)
        
        if filter_func:
            self.filters[f"{event_type}:{id(handler)}"] = filter_func
        
        self.metrics['active_subscriptions'] = self._count_subscriptions()
        self.logger.debug(f"Subscribed handler to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        # Remove from direct subscriptions
        if handler in self.subscribers.get(event_type, []):
            self.subscribers[event_type].remove(handler)
        
        if handler in self.async_subscribers.get(event_type, []):
            self.async_subscribers[event_type].remove(handler)
        
        # Remove from pattern subscriptions
        self.pattern_subscribers = [
            (pattern, h) for pattern, h in self.pattern_subscribers
            if not (pattern == event_type and h == handler)
        ]
        
        # Remove filter if exists
        filter_key = f"{event_type}:{id(handler)}"
        if filter_key in self.filters:
            del self.filters[filter_key]
        
        self.metrics['active_subscriptions'] = self._count_subscriptions()
        self.logger.debug(f"Unsubscribed handler from {event_type}")
    
    async def publish(self, event_type: str, data: Dict[str, Any],
                     source: str = "unknown", priority: EventPriority = EventPriority.NORMAL,
                     correlation_id: Optional[str] = None, **metadata):
        """
        Publish an event
        
        Args:
            event_type: Type of event
            data: Event data
            source: Source of the event
            priority: Event priority
            correlation_id: ID for correlating related events
            **metadata: Additional metadata
        """
        event = Event(
            type=event_type,
            data=data,
            source=source,
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata
        )
        
        try:
            # Add to queue with priority
            await self.event_queue.put((priority.value, event))
            self.metrics['events_published'] += 1
            
            # Add to history
            self._add_to_history(event)
            
            self.logger.debug(f"Published event: {event_type} from {source}")
            
        except asyncio.QueueFull:
            self.metrics['events_dropped'] += 1
            self.logger.warning(f"Event queue full, dropping event: {event_type}")
    
    async def emit(self, event_type: str, data: Dict[str, Any], **kwargs):
        """Convenience method for publishing events"""
        await self.publish(event_type, data, **kwargs)
    
    async def _process_events(self):
        """Process events from the queue"""
        while self.running:
            try:
                # Get event with timeout to allow checking running status
                priority, event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                
                # Process event
                await self._deliver_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    async def _deliver_event(self, event: Event):
        """Deliver event to subscribers"""
        delivered = False
        
        # Direct subscribers
        handlers = self.subscribers.get(event.type, [])
        async_handlers = self.async_subscribers.get(event.type, [])
        
        # Pattern matching subscribers
        for pattern, handler in self.pattern_subscribers:
            if fnmatch.fnmatch(event.type, pattern):
                if asyncio.iscoroutinefunction(handler):
                    async_handlers.append(handler)
                else:
                    handlers.append(handler)
        
        # Deliver to sync handlers
        for handler in handlers:
            if self._should_deliver(event, handler):
                try:
                    handler(event)
                    delivered = True
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
        
        # Deliver to async handlers
        if async_handlers:
            tasks = []
            for handler in async_handlers:
                if self._should_deliver(event, handler):
                    tasks.append(handler(event))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Error in async event handler: {result}")
                    else:
                        delivered = True
        
        if delivered:
            self.metrics['events_delivered'] += 1
    
    def _should_deliver(self, event: Event, handler: Callable) -> bool:
        """Check if event should be delivered to handler"""
        filter_key = f"{event.type}:{id(handler)}"
        
        if filter_key in self.filters:
            filter_func = self.filters[filter_key]
            try:
                return filter_func(event)
            except Exception as e:
                self.logger.error(f"Error in event filter: {e}")
                return False
        
        return True
    
    def _add_to_history(self, event: Event):
        """Add event to history for debugging"""
        self.event_history.append(event)
        
        # Trim history if too large
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
    
    def _count_subscriptions(self) -> int:
        """Count total active subscriptions"""
        count = sum(len(handlers) for handlers in self.subscribers.values())
        count += sum(len(handlers) for handlers in self.async_subscribers.values())
        count += len(self.pattern_subscribers)
        return count
    
    async def wait_for(self, event_type: str, timeout: Optional[float] = None,
                       filter_func: Optional[Callable] = None) -> Optional[Event]:
        """
        Wait for a specific event
        
        Args:
            event_type: Event type to wait for
            timeout: Maximum time to wait
            filter_func: Optional filter for the event
        
        Returns:
            Event if received, None if timeout
        """
        future = asyncio.Future()
        
        def handler(event: Event):
            if not future.done():
                future.set_result(event)
        
        self.subscribe(event_type, handler, filter_func)
        
        try:
            event = await asyncio.wait_for(future, timeout)
            return event
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(event_type, handler)
    
    async def request_response(self, request_type: str, response_type: str,
                              data: Dict[str, Any], source: str,
                              timeout: float = 30.0) -> Optional[Event]:
        """
        Send request and wait for response (RPC-style)
        
        Args:
            request_type: Request event type
            response_type: Expected response event type
            data: Request data
            source: Request source
            timeout: Response timeout
        
        Returns:
            Response event or None if timeout
        """
        import uuid
        correlation_id = str(uuid.uuid4())
        
        # Set up response listener
        response_future = asyncio.Future()
        
        def response_filter(event: Event) -> bool:
            return event.correlation_id == correlation_id
        
        # Subscribe to response
        response_task = asyncio.create_task(
            self.wait_for(response_type, timeout, response_filter)
        )
        
        # Send request
        await self.publish(
            request_type, data, source,
            correlation_id=correlation_id
        )
        
        # Wait for response
        return await response_task
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return {
            **self.metrics,
            'queue_size': self.event_queue.qsize(),
            'max_queue_size': self.event_queue.maxsize
        }
    
    def get_history(self, event_type: Optional[str] = None,
                   source: Optional[str] = None,
                   limit: int = 10) -> List[Event]:
        """
        Get event history
        
        Args:
            event_type: Filter by event type
            source: Filter by source
            limit: Maximum events to return
        
        Returns:
            List of historical events
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        return events[-limit:]
    
    async def broadcast(self, event_type: str, data: Dict[str, Any], **kwargs):
        """Broadcast event to all subscribers (alias for publish)"""
        await self.publish(event_type, data, **kwargs)
    
    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
    
    async def subscribe_async(self, patterns: List[str]) -> AsyncIterator[Event]:
        """
        Async iterator for subscribing to multiple event patterns
        
        Args:
            patterns: List of event patterns to subscribe to
        
        Yields:
            Events matching the patterns
        """
        queue = asyncio.Queue()
        
        async def handler(event: Event):
            await queue.put(event)
        
        # Subscribe to all patterns
        for pattern in patterns:
            self.subscribe(pattern, handler)
        
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            # Unsubscribe when done
            for pattern in patterns:
                self.unsubscribe(pattern, handler)


class EventChannel:
    """
    Named channel for pub/sub communication
    """
    
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self.subscribers: Set[Callable] = set()
    
    async def publish(self, data: Dict[str, Any], **kwargs):
        """Publish to this channel"""
        await self.event_bus.publish(
            f"channel:{self.name}",
            data,
            **kwargs
        )
    
    def subscribe(self, handler: Callable):
        """Subscribe to this channel"""
        self.event_bus.subscribe(f"channel:{self.name}", handler)
        self.subscribers.add(handler)
    
    def unsubscribe(self, handler: Callable):
        """Unsubscribe from this channel"""
        self.event_bus.unsubscribe(f"channel:{self.name}", handler)
        self.subscribers.discard(handler)
    
    async def request(self, data: Dict[str, Any], timeout: float = 30.0) -> Optional[Event]:
        """Send request and wait for response on this channel"""
        return await self.event_bus.request_response(
            f"channel:{self.name}:request",
            f"channel:{self.name}:response",
            data,
            self.name,
            timeout
        )