# Anything-Counter
This demo project for detection, tracking and counting anything (car, human, etc)

## Demonstration
![Demostration](assets/readme_assets/example1.gif)

## Environment
```bash
git clone https://github.com/Dragon181/Anything-Counter.git
cd Anything-Counter
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Run counter
Before run you should change [configs](config/anything_counter).

Download some detector weights ([for example](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-detection-0202/README.md)), put them in assets and run Anything-Counter!
```bash
# from repository root
python3 -m anything_counter.main
```
If you want to run it with Docker:
```bash
docker build -t anything_counter ./
docker run --rm anything_counter
```

## Development
You can write your own [detector](anything_counter/detectors), [tracker](anything_counter/trackers) or [loader](anything_counter/loaders).

But don't forget update config for your code.
I hope it would be helpfull for you!