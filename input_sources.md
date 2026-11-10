# Input sources

dxfeeder reads open data. This file lists each class of input with its licence, a retrievable address for the licence and the data, and the attribution sentence that any output or copy of the data must carry. The package code is under the Apache License 2.0 (see `LICENSE` and `NOTICE`); the licences below cover the data only.

## OpenStreetMap

- What dxfeeder reads: streets, building footprints, substations and mapped power lines from OpenStreetMap extracts in PBF format.
- Licence: Open Data Commons Open Database License (ODbL) 1.0, https://opendatacommons.org/licenses/odbl/1-0/
- Copyright and attribution guide: https://www.openstreetmap.org/copyright
- Attribution sentence: "© OpenStreetMap contributors. Data available under the Open Database License (ODbL) 1.0."
- Shipped test data: `tests/sample_data/osm_power_lines.gpkg` is derived from OpenStreetMap and remains under ODbL 1.0 with the credit above, which its layer description also carries. A feeder built from OpenStreetMap is a derived database; if it is shared or used publicly it must be offered under ODbL 1.0 and carry the same credit.

## Geofabrik extracts

- What dxfeeder reads: regional OpenStreetMap extracts are downloaded from Geofabrik, https://download.geofabrik.de/
- Licence: the extracts are OpenStreetMap data and carry the OpenStreetMap licence above (ODbL 1.0); Geofabrik is the extract provider, not a separate licensor of the data.
- Attribution sentence: "OpenStreetMap extract provided by Geofabrik GmbH (https://download.geofabrik.de/). © OpenStreetMap contributors, ODbL 1.0."

## Statistics Canada boundary files

- What dxfeeder reads: 2021 Census boundary files, such as census divisions, census subdivisions and population centres, from https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21
- Licence: Statistics Canada Open Licence, https://www.statcan.gc.ca/en/terms-conditions/open-licence
- Attribution sentence for a copy: "Source: Statistics Canada, 2021 Census boundary files, 2021. Reproduced and distributed on an "as is" basis with the permission of Statistics Canada."
- Attribution sentence for a product built from the data: "Adapted from Statistics Canada, 2021 Census boundary files, 2021. This does not constitute an endorsement by Statistics Canada of this product."
- Shipped test data: `tests/sample_data/census_divisions.gpkg` is a cut of the 2021 census division boundary file; the first sentence above applies to it.

## CanVec

- What dxfeeder reads: CanVec power line features from Natural Resources Canada, https://open.canada.ca/data/en/dataset/8ba2aa2a-7bb9-4448-b4d7-f164409fe056
- Licence: Open Government Licence - Canada, https://open.canada.ca/en/open-government-licence-canada
- Attribution sentence: "Contains information licensed under the Open Government Licence - Canada."
- Shipped test data: `tests/sample_data/canvec_power_lines.gpkg` is a cut of CanVec power lines; the sentence above applies to it.

## SMART-DS

- What dxfeeder reads: the SMART-DS synthetic distribution feeders in OpenDSS format, used as the reference that generated feeders are compared with. The data set is not shipped.
- Data set page: https://data.openei.org/submissions/2981
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Attribution sentence: "Palmintier, B., Mateo Domingo, C., Postigo Marcos, F., Gomez San Roman, T., de Cuadra, F., Gensollen, N., Elgindy, T., and Duenas, P. (2020). SMART-DS Synthetic Electrical Network Data OpenDSS Models for SFO, GSO, and AUS. Open Energy Data Initiative (OEDI), National Renewable Energy Laboratory. https://data.openei.org/submissions/2981. Licensed under CC BY 4.0." Any change made to the data must be stated next to this sentence.
