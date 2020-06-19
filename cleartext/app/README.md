# ClearText API

## Deployment

To deploy locally and in development mode, ensure you are in the current directory and execute the following command:

```bash
FLASK_ENV=development FLASK_APP=app.py flask run
``` 

To deploy to all hosts on port 8501, ensure you are in the current directory and execute the following command:

```bash
FLASK_APP=app.py flask run --host 0.0.0.0 --port 8501
```
