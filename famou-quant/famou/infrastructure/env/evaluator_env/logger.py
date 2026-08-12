# Copyright 2026 Baidu ACG FM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Local logger module for evaluator_env

Provides a simple logging interface compatible with famou_sdk.logger
"""

import logging

# Logger name prefix
LOGGER_NAME = "famou.evaluator_env"


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Sub-module name
        
    Returns:
        logging.Logger instance
    """
    if name:
        logger_name = f"{LOGGER_NAME}.{name}"
    else:
        logger_name = LOGGER_NAME
    
    logger = logging.getLogger(logger_name)
    
    # Set default level to WARNING if not already configured
    if logger.level == logging.NOTSET and not logger.handlers:
        logger.setLevel(logging.WARNING)
        # Add NullHandler to prevent lastResort handler output
        logger.addHandler(logging.NullHandler())
    
    return logger
