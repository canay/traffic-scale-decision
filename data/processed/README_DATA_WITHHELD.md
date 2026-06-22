# Record-Level Data Withheld

This public reproducibility package intentionally does not include raw or record-level processed firewall logs.

The firewall records come from an institutional production environment and were used under authorization for academic research. That authorization is not treated as permission for unrestricted public redistribution. Even processed firewall records can expose sensitive operational structure through timestamps, zones, interfaces, ports, country or geography labels, application metadata, firewall rule names, and policy-proximal fields. These fields may also create data-protection risks under applicable privacy regulations.

The public repository therefore provides:

- analysis code and run scripts,
- aggregate benchmark outputs,
- uncertainty, conformal, and selective-classification outputs,
- derived diagnostic summaries,
- generated figures,
- citation metadata,
- and schema-level documentation.

Full reruns require authorized access to the institutional data or to an equivalently approved dataset placed locally under this directory with the expected filenames:

- `traffic_three_class.csv`
- `traffic_three_class_capped_sample.csv`

