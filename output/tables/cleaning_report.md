# Cleaning & Filtering Decisions

| Dataset | Step | Before | Dropped | After | Reason |
|---------|------|-------:|--------:|------:|--------|
| IPRESS | Keep ACTIVADO only | 20,819 | 0 | 20,819 | estado != 'ACTIVADO' — deactivated or closed facilities |
| IPRESS | Drop duplicate codigo_unico | 20,819 | 26 | 20,793 | Same facility registered more than once; kept row with coordinates |
| IPRESS | Flag invalid coordinates | 20,793 | 0 | 20,793 | Coords outside Peru bbox [lon -82.0,-68.0] [lat -19.0,0.0]; kept with geometry=None |
| SUSALUD | Drop NE non-reporters | 2,431,340 | 237,753 | 2,193,587 | total_atenciones ∈ {NE_0001, NE_0002}: facility did not report |
| SUSALUD | Drop duplicate grain rows | 2,193,587 | 502,040 | 1,691,547 | Exact duplicate on (ubigeo, anho, mes, co_ipress, sexo, edad): likely double-upload |
| SUSALUD | Aggregate to facility×year | 1,691,547 | 1,678,698 | 12,849 | Sum total_atenciones/atendidos across sexo × edad × mes per facility per year |
| SUSALUD | Aggregate to district×year | 1,691,547 | 1,686,600 | 4,947 | Sum total_atenciones/atendidos across all facilities per district per year |
| CCPP | Drop null/empty geometry | 136,587 | 0 | 136,587 | Rows where geometry is None or empty — cannot be placed on map |
| CCPP | Drop duplicate coordinates | 136,587 | 44 | 136,543 | Two or more populated centers at identical (longitud, latitud) |
| DISTRITOS | Drop duplicate UBIGEO | 1,873 | 0 | 1,873 | Same UBIGEO with multiple polygons; kept largest-area polygon |