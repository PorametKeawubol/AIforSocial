// จุดเชื่อมข้ามพืช: ชีวภัณฑ์ร่วมกัน
MATCH p=(s:PlantGuide2019Reviewed)-[r]->(t:PlantGuide2019Reviewed)
WHERE r.id IN ["bcff7af4c4a3c224507d", "ca49bd6afa3bd145e7e4", "96e135c468b2bcee35da", "5797f9d5217c651a2abc", "f05b67f47ca9f6411f5a", "befb3cb72f1e76bae5ee", "b6eea6082ffaccd1ab1a", "7474a4874f5a9bc6ed4d"]
RETURN p;
