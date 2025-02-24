from . import discussion


class SimpleDiscussionStrategyFactory:

    def __init__(self):
        pass
    def create(self):
        return SimpleDiscussionStrategy()
    
    @property
    def model_id(self):
        return "simpleA"


class SimpleDiscussionStrategy:

    def __init__(self):
        self.discussion = discussion.Discussion()

    def speak(self, content, role="user"):

        self.discussion.messages.append(discussion.Message(role,content))
        if "True or false" in content:
            # Creating a simple response that always returns True
            response_msg = discussion.Message(
                role="system",
                content="True",
            )      
        else:
            # Creating a simple response that always returns True
            response_msg = discussion.Message(
                role="system",
                content="[]",
            )  
        self.discussion.messages.append(response_msg)

        return response_msg, response_msg
