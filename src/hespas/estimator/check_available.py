# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import subprocess
from typing import Optional, Union

class EstimatorNotAvailableError(Exception):
    """
    Custom exception to mark that an Estimator isn't available (usually due to not installing optional dependencies)
    Should be thrown instead of ImportError if an optionally-installable estimator has dependencies that are not available
    """
    def __init__(self, message: str, estimator_name: str):
        """
        Custom __init__() that allows passing of the estimator name for using in collecting failure reasons

        :param message: Message string as per Exception
        :param estimator_name: Display name for the estimator that failed to import
        """
        super().__init__(message)
        self.estimator_name = estimator_name

def check_programs(estimator_name: str, programs: list[Union[str, list[str], dict[str, Union[str, list[str]]]]], default_arg: str = "-h") -> Optional[dict[str, str]]:
    """
    Convenience function for checking that external programs exist and work for the estimators that require them.
    Throws an exception if any of the programs tested fail.

    :param estimator_name: Display name of the estimator, for the exception that is raised if any of the programs fail
    :param programs: List of program names to test. If the element is a string or a list, it is assumed to be a non-conditional depedency.
                     If the element is a dict, it is assumed to be a dict of conditional dependencies with the keys representing the config
                     option names. If an element is a str, or the value of a dict is a str, this is called as the binary with the
                     default argument specified at 'default_arg'. If the element is a list, or the value of the dict is a list,
                     it is directly fed into subprocess as a list of a binary as its arguments.
    :param default_arg: Default argument to use if arguments aren't specified in the programs list. -h is chosen as a reasonable guess
    :returns: Either None if all are found or a dict containing the conditionally failed programs
    :raises EstimatorNotAvailableError: Raises EstimatorNotAvailableError if any of the unconditional programs fail (i.e.: return non-zero)
                                        or the program cannot be found
    """
    unconditional_failed = []
    conditional_failed = {}

    def check_program(run_args):
        if isinstance(run_args, str):
            run_args = [run_args, default_arg]
        try:
            subprocess.run(run_args, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return "'{}' failed with: '{}'".format(" ".join(run_args), str(e))
        return None

    for program in programs:
        if isinstance(program, dict):
            for param_name, run_params in program.items():
                result = check_program(run_params)
                if result is not None:
                    conditional_failed[param_name] = result
        else:
            result = check_program(program)
            if result is not None:
                unconditional_failed.append(result)

    if len(unconditional_failed) > 0:
        raise EstimatorNotAvailableError("Checking required programs for '{}' failed: '{}'".format(estimator_name, unconditional_failed), estimator_name)
    elif len(conditional_failed) > 0:
        return conditional_failed
    return None

