# Traffic Decision Dataset Schema

This file documents the processed dataset columns used by the Paper 2 experiments. It is schema-level documentation only; no record-level firewall data are included in the public package.

| Column | Role |
|---|---|
| `target` | Three-class modeling target: Allow, Drop, or Deny |
| `raw_action` | Direct label-source field, excluded from core feature set |
| `raw_traffic_subtype` | Direct label-source field, excluded from core feature set |
| `raw_session_end_reason` | Direct label-source field, excluded from core feature set |
| `Receive Time` | Direct/time field, excluded from core feature set |
| `Generate Time` | Direct/time field, excluded from core feature set |
| `High Res Timestamp` | Direct/time field, excluded from core feature set |
| `Type` | Vendor log-type field |
| `Application` | Application context |
| `Source Zone` | Operational zone context |
| `Destination Zone` | Operational zone context |
| `Inbound Interface` | Operational interface context |
| `Outbound Interface` | Operational interface context |
| `IP Protocol` | Transport context |
| `Source Port` | Transport/service context |
| `Destination Port` | Transport/service context |
| `Source Country` | Geography context |
| `Destination Country` | Geography context |
| `Category` | URL/application category context |
| `Bytes` | Traffic volume |
| `Bytes Sent` | Traffic volume |
| `Bytes Received` | Traffic volume |
| `Packets` | Packet count |
| `Packets Sent` | Packet count |
| `Packets Received` | Packet count |
| `Elapsed Time (sec)` | Session duration |
| `Subcategory of app` | Application metadata |
| `Category of app` | Application metadata |
| `Technology of app` | Application metadata |
| `Risk of app` | Application risk metadata |
| `SaaS of app` | Application metadata |
| `AI Traffic` | Application metadata |
| `Rule` | Policy/rule context |
| `Action Source` | Policy/action-source context |

