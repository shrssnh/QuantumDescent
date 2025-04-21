#!/bin/bash

cd ..
pip install .
cd deeph
deeph-train --config ./graphene.ini