# ollama-tools Wiki (source)

This folder contains the **source for the GitHub Wiki**. To publish it:

1. **Enable the wiki** on your GitHub repo (Settings → General → Features → Wiki).
2. **Clone the wiki repo** (GitHub creates it when you add the first page):
   ```bash
   git clone https://github.com/YOUR_ORG/ollama-tools.wiki.git
   cd ollama-tools.wiki
   ```
3. **Copy these files** from this `wiki/` folder into the cloned wiki repo.  
   `Home.md` becomes the wiki front page. `_Sidebar.md` customizes the sidebar.
4. **Commit and push:**
   ```bash
   git add .
   git commit -m "Add wiki pages"
   git push origin master
   ```

Alternatively, create the wiki by hand on GitHub using the same filenames and content from this folder.
