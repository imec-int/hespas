# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Union, Any
from dataclasses import dataclass, field
from ..utils.serialisable import Serialisable

@dataclass
class CounterEvent(Serialisable):
    """
    This class represents a single Performance Counter Event. The details of the event are not specified -
    this is intended to simply serialise an event in a generic way to allowing flexible generation of
    events from Estimator sources and analysis of events in external tools (such as extracting power information).

    This class is not intended to be used alone, but rather should be contained within a CounterEventStream.
    Users can instantiate this class and serialise/deserialise this class on its own, but it is not recommended.
    The recommended interface is use to use add_event() within CounterEventStream, and to only serialise/deserialise
    entire CounterEventStream objects.

    'timestamp' is intended to be generic enough allow even Estimators without a concept of time to generate a stream
    of events. This is why 'timeunit' is seperate from 'timestamp'. A 'timeunit' of "" (the empty string) should be
    taken as:
    "each timestamp represents a unique time quanta, with timestamps less than this one happening before
    this time quanta, equal to this timestamp happening at the same time quanta, and greater than happening after
    this time quanta"
    If the timeunit is defined to a non-empty string, it should either be a reasonable unit (like 's') or something
    like 'cycles' that can be straightforwardly understood by external tools.

    'metadata' is intentionally a (possibly empty) free dictionary to be used to capture any implementation-specific
    information that is useful for external tools to understand information related to this event. No specific default
    or required metadata is specified.

    :param event_name: Name of the event being captured.
    :param timestamp: A number representing the time that this event occurs.
    :param timeunit: A string representing the unit of time for the timestamp, or empty string (meaning generic time quanta)
    :param metadata: A (possibly empty) dict containing metadata for the counter event.
    """
    event_name: str
    timestamp: Union[str, int, float]
    timeunit: str = field(default="")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        This __post_init__ first cleans up the timestamp value after initialisation.
        Converting the float to an int if is_integer() is true just gives slightly
        cleaner serialisation. The details of this should not be important to users.

        It then ensures that timeunit and event_name are strings and metadata is a dict
        """
        if isinstance(self.timestamp, str):
            self.timestamp = float(self.timestamp)
        if isinstance(self.timestamp, float) and self.timestamp.is_integer():
            self.timestamp = int(self.timestamp)
        self.event_name = str(self.event_name)
        self.timeunit = str(self.timeunit)
        self.metadata = dict(self.metadata)

