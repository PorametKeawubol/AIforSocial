// Knowledge graph: นิทานอีสป "หมาป่าอกตัญญู"
// This script is safe to run again because it uses MERGE and stable IDs.

CREATE CONSTRAINT fable_story_id IF NOT EXISTS
FOR (story:Story) REQUIRE story.id IS UNIQUE;

CREATE CONSTRAINT fable_character_id IF NOT EXISTS
FOR (character:Character) REQUIRE character.id IS UNIQUE;

CREATE CONSTRAINT fable_event_id IF NOT EXISTS
FOR (event:Event) REQUIRE event.id IS UNIQUE;

CREATE CONSTRAINT fable_place_id IF NOT EXISTS
FOR (place:Place) REQUIRE place.id IS UNIQUE;

CREATE CONSTRAINT fable_moral_id IF NOT EXISTS
FOR (moral:Moral) REQUIRE moral.id IS UNIQUE;

// Story, characters, places, and moral.
MERGE (story:Story {id: 'fable-ungrateful-wolf'})
SET story.storyId = 'fable-ungrateful-wolf',
    story.title = 'หมาป่าอกตัญญู',
    story.source = 'นิทานอีสป'

MERGE (boy:Character {id: 'fable-ungrateful-wolf-boy'})
SET boy.storyId = story.storyId, boy.name = 'เด็กเลี้ยงแกะ', boy.role = 'ผู้ช่วยหมาป่า'

MERGE (wolf:Character {id: 'fable-ungrateful-wolf-wolf'})
SET wolf.storyId = story.storyId, wolf.name = 'หมาป่า', wolf.role = 'ผู้หลอกลวง'

MERGE (sheep:Character {id: 'fable-ungrateful-wolf-sheep'})
SET sheep.storyId = story.storyId, sheep.name = 'ฝูงแกะ', sheep.role = 'สัตว์ที่เด็กดูแล'

MERGE (mountain:Place {id: 'fable-ungrateful-wolf-mountain'})
SET mountain.storyId = story.storyId, mountain.name = 'ภูเขา'

MERGE (pit:Place {id: 'fable-ungrateful-wolf-pit'})
SET pit.storyId = story.storyId, pit.name = 'หลุม'

MERGE (moral:Moral {id: 'fable-ungrateful-wolf-moral'})
SET moral.storyId = story.storyId,
    moral.text = 'ความซื่อสัตย์จริงใจ ไม่มีเล่ห์เหลี่ยม'

// Events in chronological order.
MERGE (e1:Event {id: 'fable-ungrateful-wolf-event-1'})
SET e1.storyId = story.storyId, e1.sequence = 1,
    e1.name = 'เด็กต้อนฝูงแกะลงจากภูเขา'

MERGE (e2:Event {id: 'fable-ungrateful-wolf-event-2'})
SET e2.storyId = story.storyId, e2.sequence = 2,
    e2.name = 'เด็กได้ยินเสียงหมาป่าในหลุม'

MERGE (e3:Event {id: 'fable-ungrateful-wolf-event-3'})
SET e3.storyId = story.storyId, e3.sequence = 3,
    e3.name = 'เด็กกังวลว่าหมาป่าจะกินแกะ'

MERGE (e4:Event {id: 'fable-ungrateful-wolf-event-4'})
SET e4.storyId = story.storyId, e4.sequence = 4,
    e4.name = 'หมาป่าสัญญาว่าจะไม่ทำร้ายแกะ'

MERGE (e5:Event {id: 'fable-ungrateful-wolf-event-5'})
SET e5.storyId = story.storyId, e5.sequence = 5,
    e5.name = 'เด็กช่วยหมาป่าขึ้นจากหลุม'

MERGE (e6:Event {id: 'fable-ungrateful-wolf-event-6'})
SET e6.storyId = story.storyId, e6.sequence = 6,
    e6.name = 'หมาป่าโกหกให้เด็กจำลองการตกในหลุม'

MERGE (e7:Event {id: 'fable-ungrateful-wolf-event-7'})
SET e7.storyId = story.storyId, e7.sequence = 7,
    e7.name = 'เด็กให้หมาป่ากลับลงหลุมเพื่อจำลองเหตุการณ์'

MERGE (e8:Event {id: 'fable-ungrateful-wolf-event-8'})
SET e8.storyId = story.storyId, e8.sequence = 8,
    e8.name = 'เด็กปล่อยหมาป่าไว้ในหลุมตลอดกาล'

// Context and story structure.
MERGE (story)-[:HAS_CHARACTER]->(boy)
MERGE (story)-[:HAS_CHARACTER]->(wolf)
MERGE (story)-[:HAS_CHARACTER]->(sheep)
MERGE (story)-[:HAS_PLACE]->(mountain)
MERGE (story)-[:HAS_PLACE]->(pit)
MERGE (story)-[:TEACHES]->(moral)
MERGE (story)-[:HAS_EVENT]->(e1)
MERGE (story)-[:HAS_EVENT]->(e2)
MERGE (story)-[:HAS_EVENT]->(e3)
MERGE (story)-[:HAS_EVENT]->(e4)
MERGE (story)-[:HAS_EVENT]->(e5)
MERGE (story)-[:HAS_EVENT]->(e6)
MERGE (story)-[:HAS_EVENT]->(e7)
MERGE (story)-[:HAS_EVENT]->(e8)

// Timeline, actions, and consequences.
MERGE (e1)-[:NEXT]->(e2)
MERGE (e2)-[:NEXT]->(e3)
MERGE (e3)-[:NEXT]->(e4)
MERGE (e4)-[:NEXT]->(e5)
MERGE (e5)-[:NEXT]->(e6)
MERGE (e6)-[:NEXT]->(e7)
MERGE (e7)-[:NEXT]->(e8)

MERGE (boy)-[:PERFORMS]->(e1)
MERGE (boy)-[:PERFORMS]->(e2)
MERGE (boy)-[:PERFORMS]->(e3)
MERGE (wolf)-[:PERFORMS]->(e4)
MERGE (boy)-[:PERFORMS]->(e5)
MERGE (wolf)-[:PERFORMS]->(e6)
MERGE (boy)-[:PERFORMS]->(e7)
MERGE (boy)-[:PERFORMS]->(e8)
MERGE (e1)-[:HAPPENS_AT]->(mountain)
MERGE (e2)-[:HAPPENS_AT]->(pit)
MERGE (e5)-[:HAPPENS_AT]->(pit)
MERGE (e7)-[:HAPPENS_AT]->(pit)
MERGE (e8)-[:HAPPENS_AT]->(pit)

MERGE (wolf)-[:PROMISES_NOT_TO_HARM]->(sheep)
MERGE (boy)-[:SAVES]->(wolf)
MERGE (wolf)-[:DECEIVES]->(boy)
MERGE (boy)-[:KEEPS_IN]->(pit)
MERGE (pit)-[:CONFINES]->(wolf);

// Run this in Neo4j Browser and switch the result to Graph view.
MATCH (node {storyId: 'fable-ungrateful-wolf'})
OPTIONAL MATCH (node)-[relationship]-(connected {storyId: 'fable-ungrateful-wolf'})
RETURN node, relationship, connected;
