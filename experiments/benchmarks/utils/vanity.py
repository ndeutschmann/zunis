import sys
import logging

logger = logging.getLogger(__name__)

# Python 3.7 ensures runtime UTF-8 encoding.
# However, earlier versions do not and string encoding might depend on the locale.
# Therefore we use the nice sigma symbol only if we are using python >= 3.7

if sys.version_info >= (3, 7):
    sigma = "σ"
else:
    sigma = "sig."
