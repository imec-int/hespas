# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from __future__ import annotations
import sys
import warnings
from typing import Any, ClassVar, Optional, Union, Tuple
if sys.version_info < (3, 11):
    from typing import NoReturn as Never
else:
    from typing import Never
from dataclasses import dataclass, field
from ..utils.serialisable import Serialisable
from .event import CounterEvent

@dataclass
class CounterEventStream(Serialisable):
    """
    This class represents an entire series of Performance Counter events, and handles the serialisation and deserialisation
    of the Performance Counter JSON file format. Users should use this class to serialise and deserialise the Performance
    Counter format, as well as used an instantiated object from this class to collect events within an Estimator.
    The format of the series of Performance Counter events is intentionally kept generic, so a large set of models and
    analysis tools can use it as a common intermediary format. CounterEventStream itself contains three fields, either
    set at instantiation or through deserialisation - a version, either derived from the current implementation version
    or from the contents of the deserialised JSON; some metadata within an dict, and a list of events. Each event is
    an object of type CounterEvent, which contains an event_name, a timestamp, a timeunit, and a metadata dict itself.

    The Performance Counter file format is an intentionally high-level, generic intermediate file format to allow Estimators
    to generate a list of events that external tools can process and use in the process of things like power analysis.
    The Performance Counter file format is JSON, with the following specified fields: ::

        {
            "version": "0.0.1" <- This is the required version of the generated Performance Counter file
            "metadata": <- This is an (possibly empty) metadata object that should apply to all events
            {
                "some_optional_metadata": true,
                "not_specified": 1
            },
            "events": <- This is the list of all performance counter events
            [
                {
                    "event_name": "an_event", <- The name of the event. The meaning is intentionally not specified
                    "timestamp": 1, <- This is a time quanta for this event
                    "timeunit": "", <- Empty string here means 'some abstract ordered time quanta'
                    "metadata": {} <- This is an (possibly empty) metadata object specific to this event
                },
                ...
                {
                    "event_name": "another_event",
                    "timestamp": 1.1, <- timestamps are allowed to be floats
                    "timeunit": "us", <- When timeunit is not the empty string, it should be a real time unit
                    "metadata": <- metadata is per event and can be unique to a single one
                    {
                        "unique_to_this_event": true
                    }
                }
            ]
        }

    The **version** field is intended to ensure that if the format changes in future, that older code can be aware when
    the object is instantiated through deserialisation. It should not be set manually in instantiation by users.

    **metadata** is intentionally a (possibly empty) free dictionary to be used to capture any implementation-specific
    information that is useful for external tools to understand information related to this series of events.
    No specific default or required metadata is specified. The metadata within this dictionary should be applicable
    to all events contained within the events list.

    **events** is the list of all Performance Counter events contained within this EventStream. During deserialisation,
    this will be set at time of instantiation, but in programmtic use users should instantiate a CounterEventStream
    with only the 'metadata' field (if any exists) and use 'add_event()' to add each Performance Counter event.

    Within an event (which is implemented in the CounterEvent class), each object has the following fields:
        * timestamp: intended to be generic enough allow even Estimators without a concept of time to generate a stream
          of events. This is why 'timeunit' is seperate from 'timestamp'. A 'timeunit' of "" (the empty string) should be
          taken as: "each timestamp represents a unique time quanta, with timestamps less than this one happening before
          this time quanta, equal to this timestamp happening at the same time quanta, and greater than happening after
          this time quanta" If the timeunit is defined to a non-empty string, it should either be a reasonable unit (like 's')
          or something like 'cycles' that can be straightforwardly understood by external tools.

        * metadata: intentionally a (possibly empty) free dictionary to be used to capture any implementation-specific
          information that is useful for external tools to understand information related to this event. No specific default
          or required metadata is specified.

    **class_version** is the current implementation version of the class. This will not be serialised.
    It is used for comparison with the 'version' found in deserialisation, but otherwise should not be touched by users.

    :param version: Version of the instantiated CounterEventStream. Should not be set by Users
    :param metadata: A (possibly empty) dict containing metadata for the counter event stream
    :param events: A list of CounterEvent performance counter events. Shouldn't be set in instantiation but by 'add_event()'
    """
    class_version: ClassVar[Tuple[int, int, int]] = (0, 0, 1)

    version: str = field(default=".".join([str(x) for x in class_version]))
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[CounterEvent] = field(default_factory=list)

    def __post_init__(self):
        """
        This __post_init__ checks the version of the instantiated CounterEventStream versus the implemented
        class version, and prints a warning if the version instantiated is greater than the class version.
        This would happen when a JSON file from a newer version is loaded into an older version.

        It then instantiates all the CounterEvents within the 'events' list. If this class has been created
        from deserialisation, these should all be dictionaries, so CounterEvent.from_dict() needs to be called.
        Otherwise, if this object was instantiated directly with a list of events, it leaves them alone.
        This also has the side effect of type checking the entire list on instantiation.

        Finally, ensure that metadata is actually a dict.
        """
        version_tuple = self.__ver_str_to_tuple(str(self.version))
        if len(version_tuple) != 3:
            raise ValueError("version field must be in the form '#.#.#', not '{}'".format(self.version))
        if version_tuple > self.class_version:
            class_name = self.__class__.__name__
            warnings.warn("Version of input {} '{}' is newer than runtime {} version '{}'".format(class_name, self.version, class_name, self.__ver_tuple_to_str(self.class_version)))
        self.events = [CounterEvent.from_dict(x) if not isinstance(x, CounterEvent) else x for x in self.events]
        self.metadata = dict(self.metadata)

    def add_event(self, new_event: Optional[Union[CounterEvent, dict[str, Any]]] = None, /, **kwargs) -> Never:
        """
        This should be the standard interface for reporting Performance Counter Events. This will instantiate the
        underlying CounterEvent object, and add it to the list of events. To generate a CounterEventStream file,
        users should instantiate an object from this class and use 'add_event()' on it, then serialise the result.

        There are three main ways to use this function - using keywork arguments, passing a dict, or passing an
        instantiated CounterEvent.
        Using keyword arguments is the recommended interface as this allows both 'timeunit' and 'metadata' to be
        constructed from defaults rather then requiring all of the fields to be defined.
        Passing a dict directly calls CounterEvent.from_dict() with that dict, which uses the same pathway as
        deserialisation; and thus requires all fields to be present in the dict and does not allow 'timeunit'
        and 'metadata' to be constructed from defaults. This is supported for consistency.
        Passing an instantiated CounterEvent will directly append it to the events list. Users are recommended
        to not directly instantiate their own CounterEvent objects, as they don't have a use outside of a
        CounterEventStream but this exists as not supporting this case would be inconsistent.

        The keyword arguments to use are:
        'event_name': Name of this event. Required.
        'timestamp': The timestamp this event occured at. Required.
        'timeunit': The unit of time that the timestamp has. Optional, and the default is empty string
        'metadata': dict containing metadata for this event. Optional.

        If a dict is passed instead, the keys have the same name as the keyword arguments, but all are required.

        :param new_event: Either an instantiated CounterEvent or a dict that can be used to create one. Optional.
        :param kwargs: List of keyword arguments to use to instantiate a CounterEvent. Recommended usage.
        """
        event = None
        if new_event is None:
            event = CounterEvent(**kwargs)
        elif isinstance(new_event, CounterEvent):
            event = new_event
        else:
            event = CounterEvent.from_dict(new_event)
        self.events.append(event)

    @staticmethod
    def __ver_tuple_to_str(t: Tuple[int, int, int]) -> str:
        """
        Private class convenience method to convert the current version tuple to a string (for serialisation)
        Shouldn't be called manually.

        :param t: Version tuple to convert
        :returns: String representation of version tuple
        """
        return ".".join([str(x) for x in t])

    @staticmethod
    def __ver_str_to_tuple(s: str) -> Tuple[int, int, int]:
        """
        Private class convenience method to convert the current version string to a tuple (for comparison)
        Shouldn't be called manually.

        :param s: Version string to convert
        :returns: Version tuple
        """
        return tuple([int(x) for x in s.split(".")])
