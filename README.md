# dxfeeder

dxfeeder builds synthetic electricity distribution feeders in OpenDSS format for a North American area that the user chooses. It draws its inputs from OpenStreetMap and other open data, and it validates the feeders it builds against the SMART-DS synthetic feeder data set. The same code builds Canadian and US sites; only the inputs are country specific. The showcase site is Halifax, Nova Scotia, and the SMART-DS validation site is Austin, Texas.

The tool is under construction. It builds feeders for one site from a local OpenStreetMap extract and a GeoPackage that holds the site shape, the substations and, where mapped, the distribution lines:

- `python -m dxfeeder.build --config <site.toml>` checks each input against its SHA-256 checksum and writes `Master.dss`, `Lines.dss`, `Transformers.dss`, `Loads.dss` and `Buscoords.dss`. The docstring of `dxfeeder/build.py` describes the site file.
- `python -m dxfeeder.verify <Master.dss>` compiles and solves a feeder with OpenDSS and reports whether it is radial and energizes every load. It needs the `verify` extra (`pip install dxfeeder[verify]`).
- `dxfeeder check` runs the tests and the package checks.

Every number the generator uses lives in `settings.toml`, with its label and reason.

## Data sources and licences

The dxfeeder code is under the Apache License 2.0; see `LICENSE` and `NOTICE`. The data it reads keep their own licences: OpenStreetMap (ODbL 1.0, "© OpenStreetMap contributors"), Statistics Canada boundary files (Statistics Canada Open Licence), CanVec (Open Government Licence - Canada) and SMART-DS (CC BY 4.0). `input_sources.md` lists each input class with its licence, a retrievable address and the attribution sentence to use.
