Certain details always need to be followed.

1: Always use absolute import not relative import in python. base folder is always backend.
2: Its a astral uv project (not poetry or pip): DETAILS: https://github.com/astral-sh/uv/blob/main/README.md
3: Before using lib in the code, install the lib, go into .venv folder, find the class or function you want to use and explore the args type, error and exception such that you have enough context before using.
4: If anywhere in tasks or requirements if ever a url is mentioned, review first before moving ahead as it the context decided by root user to be reviwed before moving on.
5: When ever creating anything with releated with OpenAI (use responses api not chat completion endpoint): https://platform.openai.com/docs/api-reference/responses
6: Use the logging module in each new module creation.
7: Use async where ever its possible.
8: Use tqdm/async tqdm where there is progress need to be show.
9: If identified a lib need to be added, you can use uv add lib_name
10: Write type strict code, use pydantic.
11: When ever writing sqlite engine use SQLAlchemy