// 1. Inspect the source file before importing it.
LOAD CSV WITH HEADERS FROM 'file:///Results.csv' AS row
RETURN row
LIMIT 5;

// 2. Protect the identifiers so the import is safe to re-run.
CREATE CONSTRAINT product_id IF NOT EXISTS
FOR (product:Product) REQUIRE product.productId IS UNIQUE;

CREATE CONSTRAINT student_id IF NOT EXISTS
FOR (student:Student) REQUIRE student.studentId IS UNIQUE;

// 3. Import all 33 rows as a small graph.
// Backticks are required because one CSV header is Thai text.
LOAD CSV WITH HEADERS FROM 'file:///Results.csv' AS row
WITH row
WHERE trim(row.Productid) <> '' AND trim(row.`รหัสนักศึกษา`) <> ''
MERGE (product:Product {productId: toInteger(row.Productid)})
MERGE (student:Student {studentId: trim(row.`รหัสนักศึกษา`)})
MERGE (product)-[:ASSIGNED_TO]->(student)
RETURN count(*) AS importedRows;

// 4. Verify the imported graph.
MATCH (product:Product)-[relationship:ASSIGNED_TO]->(student:Student)
RETURN product.productId AS productId,
       student.studentId AS studentId,
       type(relationship) AS relationship
ORDER BY productId;

// 5. Summary counts for the lab report.
MATCH (node)
RETURN labels(node) AS labels, count(*) AS count
ORDER BY labels;
