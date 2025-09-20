
def clean_output(response):
    """Extract readable content from agent response"""
    if hasattr(response, 'content'):
        return response.content
    
    if isinstance(response, dict):
        # Try messages array
        if 'messages' in response and response['messages']:
            last_msg = response['messages'][-1]
            if hasattr(last_msg, 'content'):
                return last_msg.content
        
        # Try direct content
        if 'content' in response:
            return response['content']
    
    # Convert to string and clean
    text = str(response)
    
    # Remove dict noise
    if text.startswith("{'messages'"):
        # Extract content from dict representation
        start = text.find("content='") + 9
        if start > 8:
            end = text.find("'", start)
            if end > start:
                return text[start:end]
    
    return text