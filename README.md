This is an linear algebra library created for the dual purpose of learning how to make a python package, and refreshing my linear algebra understanding.

# Development Tools
## `pytest`
### Custom `pytest` markers
There are a few tests that generate random matrcies and test them against known implementations of linear algebra function.
These tests take a long time, so they are skipped by default using a custom `pytest` marker `pytest.mark.expensive`.
To run these tests 
```bash
pytest --run-expensive
```