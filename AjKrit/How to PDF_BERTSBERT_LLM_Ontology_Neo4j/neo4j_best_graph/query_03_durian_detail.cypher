// เจาะลึกทุเรียน: รากเน่าโคนเน่า
MATCH p=(s:PlantGuide2019Reviewed)-[r]->(t:PlantGuide2019Reviewed)
WHERE r.id IN ["f05b67f47ca9f6411f5a", "c713e7f97926dd0d27e1", "995bd8bca3c24d184a61", "7a6a726d87da1f2c3f49", "21883011ba608f324a6e", "c2cb26bd3f9b9b6b4b8f", "befb3cb72f1e76bae5ee", "583feef70eadbafce652", "414cbae25d494d714877", "b6eea6082ffaccd1ab1a", "7474a4874f5a9bc6ed4d", "c31043e4eb3a42bd77c9", "a920a311a9a3756c7af6", "c4c9b90694b677d986fa"]
RETURN p;
