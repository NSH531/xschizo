# -*- coding: utf-8 -*-
import re
import sys

from rsa.util import private_to_public

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(private_to_public())
