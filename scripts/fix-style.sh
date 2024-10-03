#!/bin/bash

set -Eeuo pipefail

isort src/
black src/
mypy src/