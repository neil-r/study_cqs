from . import discussion


class SimpleDiscussionStrategyFactory(discussion.DiscussionStrategyFactory):

    def __init__(self):
        pass
    def create(self):
        return SimpleDiscussionStrategy()
    
    @property
    def model_id(self):
        return "simpleA"


class SimpleDiscussionStrategy(discussion.DiscussionStrategy):

    def __init__(self):
        pass

    @property
    def model_id(self):
        return "simpleA"

    def speak(self, d, content, role="user"):

        d.messages.append(discussion.Message(role,content))
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
        d.messages.append(response_msg)

        return response_msg, response_msg
