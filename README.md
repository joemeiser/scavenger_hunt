# Hidden Gems: The Best of Bucknell (prototype)

A photo scavenger hunt that runs entirely in the phone's browser. Players follow a clue, take a photo, and the app checks it two ways:

1. **Location**: is the phone within the distance limit of the gem?
2. **Image**: does the photo look like the gem's reference photos? A small AI model (DINOv2, run with Transformers.js) turns each photo into a list of numbers and compares them.

No server, no accounts, no keys. Nothing is uploaded.

## Put it on GitHub Pages

1. Create a new public repository on GitHub.
2. Upload every file in this folder (including `.nojekyll`).
3. Go to Settings > Pages, choose "Deploy from a branch", pick `main` and `/ (root)`, and save.
4. After a minute, open `https://YOUR-USERNAME.github.io/YOUR-REPO/` on a phone.
5. To install it: Share > Add to Home Screen (iPhone) or menu > Install app (Android).

## Starter gems from the web

Tap **Load starter gems from the web**. The app searches Wikimedia Commons for openly licensed photos taken on or near campus, reads their coordinates, and builds gems from them. Credits and licences are listed under **Photo credits**. The first run downloads the image model (tens of MB), so use Wi-Fi.

Three landmarks have hand-written clues and coordinates taken from Wikipedia: Rooke Chapel, Bertrand Library and Sojka Pavilion. Other gems get a plain clue made from the photo's title. Players can skip any gem with **Skip this gem**. Web gems have no backup question, so after three failed tries the app also offers to move on.

## Your own gems

Open **Setup** while standing at a gem. Type a name, clue and backup question, tap **Record my location**, and add 3 to 5 reference photos from different angles. When you have all your gems, tap **Export gems.json** and replace `gems.json` in the repository. Every player then gets your gems without any web lookup.

## Tuning

**Settings** has the distance limit, the image similarity needed, a switch for the location check, and tester mode (shows the scores after every photo). The defaults (60 m and 0.55) are starting guesses. Walk the hunt, read the scores for right and wrong photos, and set the similarity between them. **Download test log** saves every score.

The basic matcher is a low-accuracy stand-in for checking app flow when the model cannot load. Do not judge accuracy with it.

## Files

- `index.html`, `app.js`: the app
- `gems.json`: campus location, landmarks, and (after export) your gems
- `sw.js`, `manifest.webmanifest`, `icon-*.png`: what makes it installable
