
# -*- coding: utf-8 -*-
import re
import sys

from tensorflow_estimator.python.estimator.tools.checkpoint_converter import main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
