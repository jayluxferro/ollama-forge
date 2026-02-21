# Wiki source (publish to GitHub Wiki)

The files in this folder are the **source** for the project wiki. GitHub’s Wiki tab uses a **separate repo**, so you have to push this content there once for links to work instead of “Create new file”.

## One-time setup: push to GitHub Wiki

1. **Enable the wiki** (if it’s not already):
   - Repo → **Settings** → **General** → **Features** → check **Wiki** → Save.

2. **Create the wiki repo** (if it doesn’t exist yet):
   - Open the **Wiki** tab and click **Create the first page**, or clone the wiki URL once (see below). You can create a dummy page and delete it later.

3. **Clone the wiki repo** (replace `YOUR_ORG` and `YOUR_REPO` with your GitHub org/repo):
   ```bash
   git clone https://github.com/YOUR_ORG/YOUR_REPO.wiki.git
   cd YOUR_REPO.wiki
   ```

4. **Copy the wiki files** from this folder into the clone (root of the wiki repo):
   ```bash
   # From your main repo root (where wiki/ lives):
   cp wiki/*.md YOUR_REPO.wiki/
   cd YOUR_REPO.wiki
   ```
   Or copy by hand: all `.md` files from `wiki/` (including `Home.md`, `_Sidebar.md`) into the wiki repo root. **Do not** put them in a `wiki/` subfolder inside the wiki repo.

5. **Commit and push**:
   ```bash
   git add .
   git status   # should list Home.md, _Sidebar.md, Installation.md, etc.
   git commit -m "Add wiki pages from repo wiki/"
   git push origin master
   ```
   (Some wikis use `main`; if push fails, try `git push origin main`.)

6. **Check the Wiki tab**  
   You should see **Home** and the sidebar. Clicking links (Installation, Modelfile, etc.) should open those pages, not “Create new file”.

## After that

- **Edit on GitHub:** Use “Edit” on a wiki page and save; changes go to the `.wiki` repo.
- **Edit in main repo:** Change files in this `wiki/` folder, then copy updated files into the wiki clone and push again (or script it).
