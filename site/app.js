const storiesRoot = document.querySelector("#stories");
const updated = document.querySelector("#updated");
const healthRoot = document.querySelector("#health");
const dialog = document.querySelector("#story-dialog");
const detail = document.querySelector("#story-detail");

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return response.json();
}

function sparkline(counts) {
  const max = Math.max(...counts, 1);
  return [...counts].map((count) => {
    const height = Math.max(1, Math.round((count / max) * 18));
    return `<span class="bar" style="height:${height}px" title="${count}"></span>`;
  }).join("");
}

function showStory(story) {
  detail.innerHTML = `
    <h2>${escapeHtml(story.title)}</h2>
    <p>${story.n_sources} sources · first seen ${escapeHtml(story.first_seen)}</p>
    <ul>${story.headlines.map((headline) => `
      <li><strong>${escapeHtml(headline.source_id)}</strong>
      <a href="${escapeAttribute(headline.url)}" target="_blank" rel="noreferrer">
      ${escapeHtml(headline.title)}
      </a></li>`).join("")}</ul>`;
  dialog.showModal();
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value ?? "";
  return element.innerHTML;
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

async function main() {
  try {
    const [payload, health] = await Promise.all([
      loadJson("./data/stories.json"),
      loadJson("./data/health.json").catch(() => null),
    ]);
    updated.textContent = `Last updated ${payload.generated_at} · ${payload.model_version}`;
    if (health) {
      healthRoot.textContent =
        `Feeds ${health.run.n_feeds_ok} of ${health.run.n_feeds_total}`;
    }
    storiesRoot.innerHTML = payload.stories.map((story) => `
      <article class="story">
        <button type="button" data-story="${story.story_id}">
          <span class="headline">${escapeHtml(story.title)}</span>
          <span>${story.n_sources} sources · ${escapeHtml(story.first_seen)}</span>
        </button>
        <div class="sparkline" aria-label="Hourly headline counts">
          ${sparkline(story.hourly_counts)}
        </div>
      </article>`).join("");
    const storyById = new Map(payload.stories.map((story) => [story.story_id, story]));
    storiesRoot.addEventListener("click", (event) => {
      const button = event.target.closest("[data-story]");
      if (!button) return;
      showStory(storyById.get(button.dataset.story));
    });
  } catch (error) {
    updated.textContent = "No published stories are available yet.";
  }
}

document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
main();
