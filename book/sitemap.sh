#!/bin/bash

mdbook build
static-sitemap-cli -b https://skyzh.github.io/tiny-llm -r book -f xml -o > src/sitemap.xml
static-sitemap-cli -b https://skyzh.github.io/tiny-llm -r book -f txt -o > src/sitemap.txt
