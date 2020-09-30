This folder contains raw measurement data for the 2D positioning experiment (remote positioning).
Advertisement reports are transmitted from the movable tag and received by fixed anchors.

**File naming convention**

dist.{X.Xm}X{Y.Ym}.txt
* {X.Xm} corresponds to the X distance with one decimal position
* {Y.Ym} corresponds to the Y distance with one decimal position
* Example: dist.0.5mX0.4m corresponds to location with X = 0.5m and Y=0.4m

**Contents**

The files are comma separated values. Delimiter is ','.
* Timestamp
* TagID
* Anchor0 ID
* Anchor1 ID
* Anchor2 ID
* Anchor3 ID
* Anchor0 antenna 0 channel number
* Anchor0 antenna 1 channel number
* Anchor0 antenna 2 channel number
* Anchor0 antenna 3 channel number
* Anchor0 antenna 0 RSS
* Anchor0 antenna 1 RSS
* Anchor0 antenna 2 RSS
* Anchor0 antenna 3 RSS
* Anchor1 antenna 0 channel number
* Anchor1 antenna 1 channel number
* Anchor1 antenna 2 channel number
* Anchor1 antenna 3 channel number
* Anchor1 antenna 0 RSS
* Anchor1 antenna 1 RSS
* Anchor1 antenna 2 RSS
* Anchor1 antenna 3 RSS
* Anchor2 antenna 0 channel number
* Anchor2 antenna 1 channel number
* Anchor2 antenna 2 channel number
* Anchor2 antenna 3 channel number
* Anchor2 antenna 0 RSS
* Anchor2 antenna 1 RSS
* Anchor2 antenna 2 RSS
* Anchor2 antenna 3 RSS
* Anchor3 antenna 0 channel number
* Anchor3 antenna 1 channel number
* Anchor3 antenna 2 channel number
* Anchor3 antenna 3 channel number
* Anchor3 antenna 0 RSS
* Anchor3 antenna 1 RSS
* Anchor3 antenna 2 RSS
* Anchor3 antenna 3 RSS

Note: channel numbers 0, 1, 2 correspond to advertising channels 37, 38, 39 respectively. The RSS Readings are in dBm.
