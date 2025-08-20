
### Setup

- Set environment variables:
  - `TRIPADVISOR_API_KEY`
  - `GOOGLE_MAPS_API_KEY`
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`

- Install dependencies:
  - `pip install -r requirements.txt`

### Usage

- TripAdvisor Tool example (Python):
```python
from SearchPOITool import search_tool
print(search_tool.run("top attractions in Paris within 5 km"))
```