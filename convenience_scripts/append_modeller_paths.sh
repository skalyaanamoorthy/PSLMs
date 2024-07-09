#!/bin/bash

cat <<EOT >> ./pslm/bin/activate

if [ -z \$PYTHONPATH ]
then
    export PYTHONPATH="/usr/lib/modeller10.5/modlib/:/usr/lib/modeller10.5/lib/x86_64-intel8/python3.3/"
else
    export PYTHONPATH="/usr/lib/modeller10.5/modlib/:/usr/lib/modeller10.5/lib/x86_64-intel8/python3.3/:\${PYTHONPATH}:"
fi

if [ -z \$LD_LIBRARY_PATH ]
then
    export LD_LIBRARY_PATH="/usr/lib/modeller10.5/lib/x86_64-intel8"
else
    export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:/usr/lib/modeller10.5/lib/x86_64-intel8"
fi

export MODINSTALL10v5=/usr/lib/modeller10.5/

EOT

