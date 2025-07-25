from dataclasses import dataclass
from typing import List
import json


@dataclass
class MavenParagraph:
    topic: str
    sentences:List[str]
    events:List

    @property
    def content(self) -> str:
        return " ".join(self.sentences)
    
    def title(self) -> str:
        return self.topic
    
    def to_json(self):
        return {
            "sentence":self.sentences,
            "events":self.events,
        }



def read_in_maven_event_types(file_path):

    types = set()

    # Open the JSONL file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            # Load the JSON data from the line
            data = json.loads(line)
            
            for c in data['events']:
                types.add(c["type"])
            
    return types


def read_in_dataset(file_path, event_types) -> List[MavenParagraph]:

    annotated_paragraphs = []

    # Open the JSONL file and process each line
    with open(file_path, 'r', encoding="cp1252", errors= "ignore") as file:
        for line in file:
            # Load the JSON data from the line
            data = json.loads(line)
            
            # Print the data
            #print(data)
            annotated_paragraphs.append(MavenParagraph(
                topic = data['title'],
                sentences=list(
                    c['sentence']
                    for c in data['content'] if 'sentence' in c
                ),
                events=list(
                    c
                    for c in data['events'] if c["type"] in event_types
                )
            ))
            """
            {
                "title": "2006 Pangandaran earthquake and tsunami",
                "id": "8307a6b61b84d4eea42c1dd5e6e2cdba",
                "content": [
                    {
                        "sentence": "The 2006 Pangandaran earthquake and tsunami occurred on July 17 at along a subduction zone off the coast of west and central Java, a large and densely populated island in the Indonesian archipelago.",
                        "tokens": ["The", "2006", "Pangandaran", "earthquake", "and", "tsunami", "occurred", "on", "July", "17", "at", "along", "a", "subduction", "zone", "off", "the", "coast", "of", "west", "and", "central", "Java", ",", "a", "large", "and", "densely", "populated", "island", "in", "the", "Indonesian", "archipelago", "."]
                    },
                    {
                        "sentence": "The shock had a moment magnitude of 7.7 and a maximum perceived intensity of IV (\"Light\") in Jakarta, the capital and largest city of Indonesia.",
                        "tokens": ["The", "shock", "had", "a", "moment", "magnitude", "of", "7.7", "and", "a", "maximum", "perceived", "intensity", "of", "IV", "(", "``", "Light", "''", ")", "in", "Jakarta", ",", "the", "capital", "and", "largest", "city", "of", "Indonesia", "."]
                    },
                    {
                        "sentence": "There were no direct effects of the earthquake's shaking due to its low intensity, and the large loss of life from the event was due to the resulting tsunami, which inundated a portion of the Java coast that had been unaffected by the earlier 2004 Indian Ocean earthquake and tsunami that was off the coast of Sumatra.",
                        "tokens": ["There", "were", "no", "direct", "effects", "of", "the", "earthquake", "'s", "shaking", "due", "to", "its", "low", "intensity", ",", "and", "the", "large", "loss", "of", "life", "from", "the", "event", "was", "due", "to", "the", "resulting", "tsunami", ",", "which", "inundated", "a", "portion", "of", "the", "Java", "coast", "that", "had", "been", "unaffected", "by", "the", "earlier", "2004", "Indian", "Ocean", "earthquake", "and", "tsunami", "that", "was", "off", "the", "coast", "of", "Sumatra", "."]
                    },
                    {
                        "sentence": "The July 2006 earthquake was also centered in the Indian Ocean, from the coast of Java, and had a duration of more than three minutes.",
                        "tokens": ["The", "July", "2006", "earthquake", "was", "also", "centered", "in", "the", "Indian", "Ocean", ",", "from", "the", "coast", "of", "Java", ",", "and", "had", "a", "duration", "of", "more", "than", "three", "minutes", "."]}, {"sentence": "An abnormally slow rupture at the Sunda Trench and a tsunami that was unusually strong relative to the size of the earthquake were both factors that led to it being categorized as a tsunami earthquake.", "tokens": ["An", "abnormally", "slow", "rupture", "at", "the", "Sunda", "Trench", "and", "a", "tsunami", "that", "was", "unusually", "strong", "relative", "to", "the", "size", "of", "the", "earthquake", "were", "both", "factors", "that", "led", "to", "it", "being", "categorized", "as", "a", "tsunami", "earthquake", "."]
                    },
                    {
                        "sentence": "Several thousand kilometers to the southeast, surges of several meters were observed in northwestern Australia, but in Java the tsunami runups (height above normal sea level) were typically and resulted in the deaths of more than 600 people.",
                        "tokens": ["Several", "thousand", "kilometers", "to", "the", "southeast", ",", "surges", "of", "several", "meters", "were", "observed", "in", "northwestern", "Australia", ",", "but", "in", "Java", "the", "tsunami", "runups", "(", "height", "above", "normal", "sea", "level", ")", "were", "typically", "and", "resulted", "in", "the", "deaths", "of", "more", "than", "600", "people", "."]
                    },
                    {
                        "sentence": "Other factors may have contributed to exceptionally high peak runups of on the small and mostly uninhabited island of Nusa Kambangan, just to the east of the resort town of Pangandaran, where damage was heavy and a large loss of life occurred.",
                        "tokens": ["Other", "factors", "may", "have", "contributed", "to", "exceptionally", "high", "peak", "runups", "of", "on", "the", "small", "and", "mostly", "uninhabited", "island", "of", "Nusa", "Kambangan", ",", "just", "to", "the", "east", "of", "the", "resort", "town", "of", "Pangandaran", ",", "where", "damage", "was", "heavy", "and", "a", "large", "loss", "of", "life", "occurred", "."]},
                    {
                        "sentence": "Since the shock was felt with only moderate intensity well inland, and even less so at the shore, the surge arrived with little or no warning.",
                        "tokens": ["since", "the", "shock", "was", "felt", "with", "only", "moderate", "intensity", "well", "inland", ",", "and", "even", "less", "so", "at", "the", "shore", ",", "the", "surge", "arrived", "with", "little", "or", "no", "warning", "."]
                    },
                    {
                        "sentence": "Other factors contributed to the tsunami being largely undetected until it was too late and, although a tsunami watch was posted by an American tsunami warning center and a Japanese meteorological center, no information was delivered to people at the coast.",
                        "tokens": ["Other", "factors", "contributed", "to", "the", "tsunami", "being", "largely", "undetected", "until", "it", "was", "too", "late", "and", ",", "although", "a", "tsunami", "watch", "was", "posted", "by", "an", "American", "tsunami", "warning", "center", "and", "a", "Japanese", "meteorological", "center", ",", "no", "information", "was", "delivered", "to", "people", "at", "the", "coast", "."]
                    }
                    ],
                    "events": [
                        {
                            "id": "40b3b20bc2eeb6b163538b82c1379ead",
                            "type": "Know",
                            "type_id": 1,
                            "mention": [
                                {
                                    "trigger_word": "observed",
                                    "sent_id": 5,
                                    "offset": [12, 13],
                                    "id": "7fcf445a679aa13511278d321a908bd2"
                                }
                            ]
                        },
                        {
                            "id": "e5fe210baa4cee8d7416ea70029f5dca",
                            "type": "Warning",
                            "type_id": 2,
                            "mention": [
                                {
                                    "trigger_word": "warning",
                                    "sent_id": 7,
                                    "offset": [27, 28],
                                    "id": "e44589211d4484950c4638552129a690"
                                }
                            ]
                        }, {"id": "c4e66add7137585d164f18a0274d84c5", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "earthquake", "sent_id": 2, "offset": [50, 51], "id": "362ab206406caeb847173ca9376da937"}]}, {"id": "c5b9506f10e1a5161b159936f376f8a1", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "earthquake", "sent_id": 3, "offset": [3, 4], "id": "e6f7f136e1d75b60493c37ff4f0a1871"}]}, {"id": "fa8170feadff6b9471c0cc2eb052027f", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "earthquake", "sent_id": 2, "offset": [7, 8], "id": "df2bea73a9a9fa25cd18d424e5ae4baf"}]}, {"id": "941e90cbd133980f866a3454aa1d7892", "type": "Placing", "mention": [{"trigger_word": "centered", "sent_id": 3, "offset": [6, 7], "id": "4cec3a7de5f6404b68cac3a6db6fc19f"}], "type_id": 4}, {"id": "4bb6e5779814b7f7b7e61daf8cba7edb", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "earthquake", "sent_id": 4, "offset": [21, 22], "id": "50b66b46ce5dd8f4131f440977249085"}]}, {"id": "2f25703efe8cfabaa86f6daf92b1dfd1", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "earthquake", "sent_id": 4, "offset": [34, 35], "id": "daa4c8e50ff4f58d792113858be1fa91"}]}, {"id": "2df38ddb86831f430928116475352d8f", "type": "Causation", "type_id": 5, "mention": [{"trigger_word": "resulted in", "sent_id": 5, "offset": [33, 35], "id": "a4706465c333e8a5720bb4fd829d7646"}]}, {"id": "40f3e019d518832a15446f140b5947fc", "type": "Arriving", "type_id": 6, "mention": [{"trigger_word": "arrived", "sent_id": 7, "offset": [22, 23], "id": "efcbdad6767e213d4b3cc0390942aff1"}]}, {"id": "b06f2a463c98b3c9026901ab21b01495", "type": "Sending", "type_id": 7, "mention": [{"trigger_word": "posted", "sent_id": 8, "offset": [21, 22], "id": "591ed4ca59b2118dd39f60c2d9e22b43"}]}, {"id": "502c5d7174f792cf8bbef893bb2302b9", "type": "Sending", "mention": [{"trigger_word": "delivered", "sent_id": 8, "offset": [37, 38], "id": "18f156905d84452fdf16b2b00583b760"}], "type_id": 7}, {"id": "e5aecc775871eeea56e16bcec840df2e", "type": "Causation", "type_id": 5, "mention": [{"trigger_word": "due", "sent_id": 2, "offset": [26, 27], "id": "6c5aa79db4c42ee34ceae42586b483ae"}]}, {"id": "060b6ff06afba025990ce2cb252e81a4", "type": "Causation", "type_id": 5, "mention": [{"trigger_word": "unaffected", "sent_id": 2, "offset": [43, 44], "id": "7d9b68c67744027272060adffb06c15f"}]}, {"id": "c2465715ba2dbed5b3fcd5631c46217f", "type": "Protest", "type_id": 8, "mention": [{"trigger_word": "relative", "sent_id": 4, "offset": [15, 16], "id": "7313e32bdbd64bb2df992fc5c5f8f42c"}]}, {"id": "6c4f6df19e5430913415faebb764b969", "type": "Preventing_or_letting", "type_id": 9, "mention": [{"trigger_word": "undetected", "sent_id": 8, "offset": [8, 9], "id": "5a084ae7bba66e7f05017a826cee7638"}]}, {"id": "5dcc772f6296c122f00b67239fa712cd", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "earthquake", "sent_id": 0, "offset": [3, 4], "id": "1594c4acbc1a8a86554ac030ce128dd9"}]}, {"id": "4e92c3a20df9eb3cdf8769f67df51d6a", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "shock", "sent_id": 1, "offset": [1, 2], "id": "5ef56a9668fac25a04490726030219b2"}]}, {"id": "45211b9f669210328005edb6a18fb9f6", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "tsunami", "sent_id": 2, "offset": [30, 31], "id": "427bb797855493e6164e7f300e1d3be6"}, {"trigger_word": "tsunami", "sent_id": 0, "offset": [5, 6], "id": "6e89dee6e1778e7697915d88f4a5feae"}]}, {"id": "73ebd17b211a48bb2180c56f590e6deb", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "tsunami", "sent_id": 4, "offset": [10, 11], "id": "a7ff8a82db2345b7fb07f6de91fa2302"}]}, {"id": "f577a21baa679b3272b10f22e84d6934", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "tsunami", "sent_id": 4, "offset": [33, 34], "id": "91a4d7592e086efdff2fb0f8f3b43fb5"}]}, {"id": "a23139ca7f23e3cb8843e4c162afa50f", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "tsunami", "sent_id": 5, "offset": [21, 22], "id": "03c5a5a0052bcd4e0b245a3d5e5b393c"}]}, {"id": "dccd67dff895157703c3d86b6dc942c8", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "tsunami", "sent_id": 8, "offset": [5, 6], "id": "8dce85ed8a5b59f58c0e699e45654b08"}]}, {"id": "cc9524669cc574d5bc6d84f9a3f91c7b", "type": "Motion", "mention": [{"trigger_word": "shaking", "sent_id": 2, "offset": [9, 10], "id": "27f7fcdfce9283ca847c672e16ab82d8"}], "type_id": 10}, {"id": "4c8b06163a99a77ecf50f596c407fd23", "type": "Damaging", "type_id": 11, "mention": [{"trigger_word": "loss", "sent_id": 2, "offset": [19, 20], "id": "52fc7c45c83ffc1691c38911e15dd3f2"}]}, {"id": "16216881a9accf2899b931dd180c232a", "type": "Causation", "type_id": 5, "mention": [{"trigger_word": "resulting", "sent_id": 2, "offset": [29, 30], "id": "804da7a3f324aa12f5de8968489e396e"}]}, {"id": "078a83788cfecefb8d79824294cd8bba", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "tsunami", "sent_id": 2, "offset": [52, 53], "id": "3f5790afdded237fb044c71f66ef1487"}]}, {"id": "542a1a4d9c57b976d1dce4e56b305a4f", "type": "Destroying", "type_id": 12, "mention": [{"trigger_word": "rupture", "sent_id": 4, "offset": [3, 4], "id": "39dd632ac1917377909f7839e918907a"}]}, {"id": "9486ec807690715bc6721d6261d8225a", "type": "Causation", "type_id": 5, "mention": [{"trigger_word": "led to", "sent_id": 4, "offset": [26, 28], "id": "249bd4c4c9174dd21cae790a6922abb3"}]}, {"id": "8c95bd3f282d4b33401bb9c78f03f094", "type": "Death", "type_id": 13, "mention": [{"trigger_word": "deaths", "sent_id": 5, "offset": [36, 37], "id": "3c9285980c60daf781b8d82118881804"}]}, {"id": "28104668a13b5c8b186465451ae2e7fc", "type": "Damaging", "type_id": 11, "mention": [{"trigger_word": "damage", "sent_id": 6, "offset": [34, 35], "id": "0c106203df531c0b5179c6a5966abe7e"}]}, {"id": "8146babefa88c20314c95e74bd4ea49b", "type": "Death", "type_id": 13, "mention": [{"trigger_word": "loss", "sent_id": 6, "offset": [40, 41], "id": "afc59df76ac5043169c27e63574e47d0"}]}, {"id": "700cf6339f9be47438ed25cbcd11323f", "type": "Catastrophe", "type_id": 3, "mention": [{"trigger_word": "shock", "sent_id": 7, "offset": [2, 3], "id": "4e8bdfe4b3d9851e2bfa33b7da97dad8"}]}, {"id": "644aabe732edd7b80615b0ae365a3859", "type": "Perception_active", "type_id": 14, "mention": [{"trigger_word": "felt", "sent_id": 7, "offset": [4, 5], "id": "44f9f9f7855d6b09751ccca86dd74c4a"}]}, {"id": "6fc1baeafb2ad2a7ce8f37a6eb4f2268", "type": "Presence", "type_id": 15, "mention": [{"trigger_word": "occurred", "sent_id": 0, "offset": [6, 7], "id": "4cb375ec3c8133309c697420d323aadf"}]}, {"id": "00deace7b3607024465d9c11b02c9175", "type": "Know", "type_id": 1, "mention": [{"trigger_word": "perceived", "sent_id": 1, "offset": [11, 12], "id": "861260ae0399d3b05e026f6eae198773"}]}, {"id": "22ad48542396b5714eb3b181262c9116", "type": "Influence", "type_id": 16, "mention": [{"trigger_word": "effects", "sent_id": 2, "offset": [4, 5], "id": "850a7041c5582037799073b76c1a549b"}]}, {"id": "f5edad41f26cdc776b7d1a6f1c8609b0", "type": "Destroying", "type_id": 12, "mention": [{"trigger_word": "inundated", "sent_id": 2, "offset": [33, 34], "id": "5e16870c1316a6b26d83508a600ead6a"}]}, {"id": "d569e937ab0547524a90115eb20a1da7", "type": "Presence", "type_id": 15, "mention": [{"trigger_word": "occurred", "sent_id": 6, "offset": [43, 44], "id": "ffda4080808c5f6d65fbd672119b5343"}]}], "negative_triggers": [{"trigger_word": "populated", "sent_id": 0, "offset": [28, 29], "id": "0d871abd0017f444afc10c4e5c238f8d"}, {"trigger_word": "surges", "sent_id": 5, "offset": [7, 8], "id": "c5494687e5e52e9dd36ebecb12a9053c"}, {"trigger_word": "event", "sent_id": 2, "offset": [24, 25], "id": "13f91440d95426ff26c93d20fe26aca2"}, {"trigger_word": "surge", "sent_id": 7, "offset": [21, 22], "id": "e345c880f98975154ac82c2fd5c792e3"}, {"trigger_word": "Pangandaran", "sent_id": 0, "offset": [2, 3], "id": "ec7e4e2a067b01b0f195fea238f30a8e"}, {"trigger_word": "July", "sent_id": 0, "offset": [8, 9], "id": "5c8ec73bf52b13fec59814fdf9af9c92"}, {"trigger_word": "subduction", "sent_id": 0, "offset": [13, 14], "id": "9118d801b5dabba183d67a8d00ed6170"}, {"trigger_word": "zone", "sent_id": 0, "offset": [14, 15], "id": "bd920dbd8000498de1636b64b5113266"}, {"trigger_word": "coast", "sent_id": 0, "offset": [17, 18], "id": "db113d6975e3a1bbde1fd7837f7dd638"}, {"trigger_word": "west", "sent_id": 0, "offset": [19, 20], "id": "d818d39ba99aa7ad059413562bedb92b"}, {"trigger_word": "central", "sent_id": 0, "offset": [21, 22], "id": "f4e683d96ca99e84f546d0c4291ffc5a"}, {"trigger_word": "Java", "sent_id": 0, "offset": [22, 23], "id": "02ef8e1102f0636581966f6c24116657"}, {"trigger_word": "large", "sent_id": 0, "offset": [25, 26], "id": "bf8b2f00960e0abdfb7a7e3509d55404"}, {"trigger_word": "densely", "sent_id": 0, "offset": [27, 28], "id": "73e5365470db25500872e9c1961fdd57"}, {"trigger_word": "island", "sent_id": 0, "offset": [29, 30], "id": "4211c8df31d79a27d0523057d5aaf858"}, {"trigger_word": "Indonesian", "sent_id": 0, "offset": [32, 33], "id": "07ac142df6bfe92b44ff71131646ea42"}, {"trigger_word": "archipelago", "sent_id": 0, "offset": [33, 34], "id": "21d2e21e5beb52a99f9e114902b9b9d6"}, {"trigger_word": "moment", "sent_id": 1, "offset": [4, 5], "id": "ebd039d6c23bf7ee36770321e57d4a4d"}, {"trigger_word": "magnitude", "sent_id": 1, "offset": [5, 6], "id": "5b67fe844ac73041cd206cd223895c7b"}, {"trigger_word": "maximum", "sent_id": 1, "offset": [10, 11], "id": "0506bd88318d1d55bdbb3b95d90011ff"}, {"trigger_word": "intensity", "sent_id": 1, "offset": [12, 13], "id": "8e96a5be2126cba5648db13e8312becd"}, {"trigger_word": "IV", "sent_id": 1, "offset": [14, 15], "id": "5c3227376a34075499e34f40b338af66"}, {"trigger_word": "Light", "sent_id": 1, "offset": [17, 18], "id": "40857111764dc783bf6b9f1c4ae1d39a"}, {"trigger_word": "Jakarta", "sent_id": 1, "offset": [21, 22], "id": "c82abc8e39e144c4053508979d6afaa3"}, {"trigger_word": "capital", "sent_id": 1, "offset": [24, 25], "id": "0e707e69f9d90094b4dcd60e10e3eec8"}, {"trigger_word": "largest", "sent_id": 1, "offset": [26, 27], "id": "d4a6f92aee18555b08ee4bb87192c998"}, {"trigger_word": "city", "sent_id": 1, "offset": [27, 28], "id": "78fde33145394f8bc0a803a53e3387d1"}, {"trigger_word": "Indonesia", "sent_id": 1, "offset": [29, 30], "id": "a98e031c95181a84d1396dd3e9e14eea"}, {"trigger_word": "direct", "sent_id": 2, "offset": [3, 4], "id": "24bfd54cf599ac616510e6c2e21d1f25"}, {"trigger_word": "due", "sent_id": 2, "offset": [10, 11], "id": "226a07fb19b9749bd97bc41876c37cf0"}, {"trigger_word": "low", "sent_id": 2, "offset": [13, 14], "id": "58e72e99d19f32e46b9ba78970609d78"}, {"trigger_word": "intensity", "sent_id": 2, "offset": [14, 15], "id": "d75cdde6073f2f482c496c91819a035f"}, {"trigger_word": "large", "sent_id": 2, "offset": [18, 19], "id": "278fe8124eee57a11df8dde7893330dd"}, {"trigger_word": "life", "sent_id": 2, "offset": [21, 22], "id": "adfdc47575de607a2ec1aa38b17658bc"}, {"trigger_word": "portion", "sent_id": 2, "offset": [35, 36], "id": "704687438117f73201efbed0ebd4318d"}, {"trigger_word": "Java", "sent_id": 2, "offset": [38, 39], "id": "bc3e30aeec2fbd01302c226335716bde"}, {"trigger_word": "coast", "sent_id": 2, "offset": [39, 40], "id": "6efc5aea24b6e4d5c9899269683591ca"}, {"trigger_word": "earlier", "sent_id": 2, "offset": [46, 47], "id": "27ee5cc6dbb3b96d78a6f9496dc53931"}, {"trigger_word": "Indian", "sent_id": 2, "offset": [48, 49], "id": "907b2cd6f81ad3b480422e9d58f0d42a"}, {"trigger_word": "Ocean", "sent_id": 2, "offset": [49, 50], "id": "ed06a82dd06cf6742e4f5fbab3b25181"}, {"trigger_word": "coast", "sent_id": 2, "offset": [57, 58], "id": "c3c2cbfa6e4c8c08577280e6334c3d97"}, {"trigger_word": "Sumatra", "sent_id": 2, "offset": [59, 60], "id": "e0cc8f292575b7902237b75396c33d12"}, {"trigger_word": "July", "sent_id": 3, "offset": [1, 2], "id": "b91d2a82884415900bd3d18981b7276c"}, {"trigger_word": "also", "sent_id": 3, "offset": [5, 6], "id": "c7602417874c96b337bb8c1b3acf932e"}, {"trigger_word": "Indian", "sent_id": 3, "offset": [9, 10], "id": "96464ca3e7813461772d458a9bd89b5e"}, {"trigger_word": "Ocean", "sent_id": 3, "offset": [10, 11], "id": "da3e9109540eb3d287007fe4079de44d"}, {"trigger_word": "coast", "sent_id": 3, "offset": [14, 15], "id": "0dd8fc371890a9de88c615b82ad7366f"}, {"trigger_word": "Java", "sent_id": 3, "offset": [16, 17], "id": "ea1a7d85723aec1eb388d9727e53806e"}, {"trigger_word": "duration", "sent_id": 3, "offset": [21, 22], "id": "c7417457094ff939cfb5a0768ad9b525"}, {"trigger_word": "minutes", "sent_id": 3, "offset": [26, 27], "id": "9c34e54ba5f100110e147b50f20ac9e3"}, {"trigger_word": "abnormally", "sent_id": 4, "offset": [1, 2], "id": "2031f87a82547f6b7a0c1b1136ecfe17"}, {"trigger_word": "slow", "sent_id": 4, "offset": [2, 3], "id": "4e3ae0981b7c37857739cbbd43d1f151"}, {"trigger_word": "Sunda", "sent_id": 4, "offset": [6, 7], "id": "be927e4776daf41ba84245b64af89b5b"}, {"trigger_word": "Trench", "sent_id": 4, "offset": [7, 8], "id": "09c6ccd4ac1aa6ae74726a249993790d"}, {"trigger_word": "unusually", "sent_id": 4, "offset": [13, 14], "id": "b07f645d0e07f190a40694b7eb3d244b"}, {"trigger_word": "strong", "sent_id": 4, "offset": [14, 15], "id": "76603bc80b096a12b8a6d238bfcf7387"}, {"trigger_word": "size", "sent_id": 4, "offset": [18, 19], "id": "c4189335cf873fbb39afc16ac0148365"}, {"trigger_word": "factors", "sent_id": 4, "offset": [24, 25], "id": "4e10fc024c0017a40069073a5e39ec5e"}, {"trigger_word": "categorized", "sent_id": 4, "offset": [30, 31], "id": "1e753a8893f4cac8097023d74b46a561"}, {"trigger_word": "Several", "sent_id": 5, "offset": [0, 1], "id": "426dbfccc46697eeb0905b1106a085c9"}, {"trigger_word": "kilometers", "sent_id": 5, "offset": [2, 3], "id": "d5907c5e4da4ad097abf82f3c3ec1a92"}, {"trigger_word": "southeast", "sent_id": 5, "offset": [5, 6], "id": "e5acc4b6aae9f109905b923a07d20ab6"}, {"trigger_word": "several", "sent_id": 5, "offset": [9, 10], "id": "a572d32db28df5426c516c59344fb8d0"}, {"trigger_word": "meters", "sent_id": 5, "offset": [10, 11], "id": "47d40d3ce2224823dcd4759e274b8d5b"}, {"trigger_word": "northwestern", "sent_id": 5, "offset": [14, 15], "id": "6288553aa7f9da677d9e09c5eeebc67d"}, {"trigger_word": "Australia", "sent_id": 5, "offset": [15, 16], "id": "36eb79b1b03c1cb82cfd185f34bb2e36"}, {"trigger_word": "Java", "sent_id": 5, "offset": [19, 20], "id": "6b67842a97f89e738d50067ef1a179ab"}, {"trigger_word": "runups", "sent_id": 5, "offset": [22, 23], "id": "1647c6ad34fb1436e81898c777de3773"}, {"trigger_word": "height", "sent_id": 5, "offset": [24, 25], "id": "443f41103c54234548bdc796efbdaa4f"}, {"trigger_word": "normal", "sent_id": 5, "offset": [26, 27], "id": "a6bf499e3a944833dcd1bb09fccbbc0b"}, {"trigger_word": "sea", "sent_id": 5, "offset": [27, 28], "id": "17cfbc0b96a89c91ad386bdc04cf19b5"}, {"trigger_word": "level", "sent_id": 5, "offset": [28, 29], "id": "8416351973fcd78249711339757ba6d2"}, {"trigger_word": "typically", "sent_id": 5, "offset": [31, 32], "id": "049bf52dffb3035a5d5b7d8b4571e7c6"}, {"trigger_word": "people", "sent_id": 5, "offset": [41, 42], "id": "a18599219a3e536dd9af8e51b0aa09d9"}, {"trigger_word": "Other", "sent_id": 6, "offset": [0, 1], "id": "5afabec6de2a85433caa70e3a24ae2cc"}, {"trigger_word": "factors", "sent_id": 6, "offset": [1, 2], "id": "9cbc4adc08e6214f5942c879055cb78f"}, {"trigger_word": "exceptionally", "sent_id": 6, "offset": [6, 7], "id": "e9c2b633daab1871755f3922738e3a7b"}, {"trigger_word": "high", "sent_id": 6, "offset": [7, 8], "id": "dc226625f9b8a7c866817968337ffd28"}, {"trigger_word": "peak", "sent_id": 6, "offset": [8, 9], "id": "0c83721304090d3e761e0e47dc079a46"}, {"trigger_word": "runups", "sent_id": 6, "offset": [9, 10], "id": "1cd065cf376b4ab401aa6c8b8dde528f"}, {"trigger_word": "small", "sent_id": 6, "offset": [13, 14], "id": "0284a0672f1d01868aa4a53f0665cc79"}, {"trigger_word": "mostly", "sent_id": 6, "offset": [15, 16], "id": "c74cb8c5ffa408939e317de7050ec2af"}, {"trigger_word": "uninhabited", "sent_id": 6, "offset": [16, 17], "id": "a208b974c6734675ec3551d259f48d5d"}, {"trigger_word": "island", "sent_id": 6, "offset": [17, 18], "id": "a719b0b813005e9d912d9f27d6dc8d3d"}, {"trigger_word": "Nusa", "sent_id": 6, "offset": [19, 20], "id": "449fe25c5ee58dc02782fadc43fa86d0"}, {"trigger_word": "Kambangan", "sent_id": 6, "offset": [20, 21], "id": "56c4f721b5a5f5ad913c3965da979fe6"}, {"trigger_word": "east", "sent_id": 6, "offset": [25, 26], "id": "50d5a07fd95dcc62eed77126aca82615"}, {"trigger_word": "resort", "sent_id": 6, "offset": [28, 29], "id": "7a21665b94fec4a11d023447a4bc9edd"}, {"trigger_word": "town", "sent_id": 6, "offset": [29, 30], "id": "e212faf4196044b3a569de9d0fdd5dce"}, {"trigger_word": "Pangandaran", "sent_id": 6, "offset": [31, 32], "id": "d68602cb105c800f276aa6820a4dd859"}, {"trigger_word": "heavy", "sent_id": 6, "offset": [36, 37], "id": "de02e2deeb6ee7cfc66f7b29db8e30f5"}, {"trigger_word": "large", "sent_id": 6, "offset": [39, 40], "id": "029b688d6a40b95d54b61b06775a53b1"}, {"trigger_word": "life", "sent_id": 6, "offset": [42, 43], "id": "8d3bfddcb37d0c1186ce7b2b4f1915fb"}, {"trigger_word": "moderate", "sent_id": 7, "offset": [7, 8], "id": "833d18c1efb05b8ddcd39b319964e328"}, {"trigger_word": "intensity", "sent_id": 7, "offset": [8, 9], "id": "aad344dc7d47eb9f0183d2551b880b63"}, {"trigger_word": "well", "sent_id": 7, "offset": [9, 10], "id": "6eb3fb007bf2901be57d770cfb9a367a"}, {"trigger_word": "inland", "sent_id": 7, "offset": [10, 11], "id": "ca413d33a6b4983eb9f15d73255e5b37"}, {"trigger_word": "even", "sent_id": 7, "offset": [13, 14], "id": "03b3f0337e60498dc31c73249d99fdcb"}, {"trigger_word": "less", "sent_id": 7, "offset": [14, 15], "id": "5263dd10a8a6d26377c36fc9d9dc1b7c"}, {"trigger_word": "shore", "sent_id": 7, "offset": [18, 19], "id": "662e05360e73057831c9727af282043d"}, {"trigger_word": "little", "sent_id": 7, "offset": [24, 25], "id": "c93e2446134b1d47ebaa7150cf80d595"}, {"trigger_word": "Other", "sent_id": 8, "offset": [0, 1], "id": "28d342d0d29ff2059a1a3053a6cd1256"}, {"trigger_word": "factors", "sent_id": 8, "offset": [1, 2], "id": "af66e9acd104da95d0419fe065710283"}, {"trigger_word": "largely", "sent_id": 8, "offset": [7, 8], "id": "b85264a80ddfcd4934a103b0697bf25d"}, {"trigger_word": "late", "sent_id": 8, "offset": [13, 14], "id": "1f059ef7926a510a9418900aea5c0f81"}, {"trigger_word": "tsunami", "sent_id": 8, "offset": [18, 19], "id": "36bd042942cf2087ed92a6ed6c4b7f6c"}, {"trigger_word": "watch", "sent_id": 8, "offset": [19, 20], "id": "e58d062f26dc938fd40b0928cc5b519f"}, {"trigger_word": "American", "sent_id": 8, "offset": [24, 25], "id": "73fe2fc93fde617347c9c4b57c544ee7"}, {"trigger_word": "tsunami", "sent_id": 8, "offset": [25, 26], "id": "eba595f013084d4788741b9e300b092a"}, {"trigger_word": "warning", "sent_id": 8, "offset": [26, 27], "id": "6f3b2bf0dc95c3a3c10a564660637efe"}, {"trigger_word": "center", "sent_id": 8, "offset": [27, 28], "id": "e2463e665463a1399f07ace2b28d31e0"}, {"trigger_word": "Japanese", "sent_id": 8, "offset": [30, 31], "id": "bf1bfb7e39a42deaeb96818a96f84ba1"}, {"trigger_word": "meteorological", "sent_id": 8, "offset": [31, 32], "id": "f9687d33491eed0d2962f37b3df83ddf"}, {"trigger_word": "center", "sent_id": 8, "offset": [32, 33], "id": "3465903d7a29a452d21d0bebdc74503b"}, {"trigger_word": "information", "sent_id": 8, "offset": [35, 36], "id": "17c7c7a0609e031a989a8c61d9126c77"}, {"trigger_word": "people", "sent_id": 8, "offset": [39, 40], "id": "d8c0530f32ba636de8221c1268796cb0"}, {"trigger_word": "coast", "sent_id": 8, "offset": [42, 43], "id": "e5e0e0e29616a475c4773029d3c8a839"}]}

            """
            # annotated_paragraphs.append(MavenParagraph(
            #     sentences=list(
            #         c['sentence']
            #         for c in data['content'] if 'sentence' in c
            #     ),
            #     events=list(
            #         c
            #         for c in data['events'] if c["type"] in types
            #     )
            # ))
            #return

            # Process the data as needed
            # For example, accessing specific fields:
            # print(data['field_name'])
    
    return annotated_paragraphs



if __name__ == "__main__":
    print(len(list(r for r in read_in_dataset("data/maven/train.jsonl", event_types=["Attack"]) if len(r.events)> 0 )))

    print(read_in_maven_event_types("data/maven/train.jsonl"))
