export default {
  async scheduled(event, env, ctx) {
    const owner = "Chriz76";
    const repo = "weather-data";
    const pipelineBranch = "main";
    const dataBranch = "gh-pages";
    const workflowFileName = "sync.yml";

    const githubHeaders = {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${env.GH_PAT}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "Cloudflare-Worker"
    };

    // AROME / Open-Meteo S3 Konfiguration
    const modelName = "meteofrance_arome_france0025_15min";
    const hoursToCheck = 6; // erweiterter Suchfenster-Bereich
    const forecastOffsetHours = 6; // Offset für die letzte Datei (+6 Stunden)

    // SCHRITT 1: Prüfen, ob die Pipeline aktuell bereits läuft
    const workflowUrl = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflowFileName}/runs?status=in_progress&branch=${pipelineBranch}`;
    const workflowCheck = await fetch(workflowUrl, { headers: githubHeaders });
    if (workflowCheck.ok) {
      const runData = await workflowCheck.json();
      if (runData.total_count > 0) {
        console.log("Pipeline läuft bereits. Dieser Durchlauf wird übersprungen.");
        return;
      }
    }

    // SCHRITT 2: Neuesten VOLLSTÄNDIGEN AROME-Run (Open-Meteo S3) ermitteln
    const utcNow = new Date(new Date().toUTCString());
    let targetTime = null;
    let targetRunString = "";

    for (let i = 0; i < hoursToCheck; i++) {
      const checkTime = new Date(utcNow.getTime() - i * 60 * 60 * 1000);
      const YYYY = checkTime.getUTCFullYear();
      const MM = String(checkTime.getUTCMonth() + 1).padStart(2, '0');
      const DD = String(checkTime.getUTCDate()).padStart(2, '0');
      const HH = String(checkTime.getUTCHours()).padStart(2, '0');

      // Ordnerpfad auf S3 (Start des Runs)
      const runHourDir = `${HH}00Z`;
      const folderStr = `${YYYY}-${MM}-${DD}T${HH}:00`;

      // Ziel-Datei auf S3 ermitteln (Run-Startzeit + 6 Stunden)
      const s3TargetTime = new Date(checkTime.getTime() + forecastOffsetHours * 60 * 60 * 1000);
      const s3YYYY = s3TargetTime.getUTCFullYear();
      const s3MM = String(s3TargetTime.getUTCMonth() + 1).padStart(2, '0');
      const s3DD = String(s3TargetTime.getUTCDate()).padStart(2, '0');
      const s3HH = String(s3TargetTime.getUTCHours()).padStart(2, '0');

      // Dateiname der letzten erwarteten S3-Datei (z.B. 2026-08-06T0200.om)
      const isoFile = `${s3YYYY}-${s3MM}-${s3DD}T${s3HH}00.om`;
      const s3Url = `https://openmeteo.s3.amazonaws.com/data_spatial/${modelName}/${YYYY}/${MM}/${DD}/${runHourDir}/${isoFile}`;

      const headResp = await fetch(s3Url, { method: "HEAD" });
      if (headResp.status === 200) {
        targetTime = checkTime;
        targetRunString = folderStr;
        console.log(`🎯 Neuer vollständiger AROME-Run (inkl. ${isoFile}) gefunden: ${targetRunString}Z`);
        break;
      }
    }

    if (!targetTime) {
      console.log("❌ Kein vollständiger AROME-Datensatz im Abfragefenster gefunden.");
      return;
    }

    // SCHRITT 3: Prüfen, ob das finale WebP bereits im Repo existiert (Startzeit + 6h)
    const pngTime = new Date(targetTime.getTime() + forecastOffsetHours * 60 * 60 * 1000);
    const pYYYY = pngTime.getUTCFullYear();
    const pMM = String(pngTime.getUTCMonth() + 1).padStart(2, '0');
    const pDD = String(pngTime.getUTCDate()).padStart(2, '0');
    const pHH = String(pngTime.getUTCHours()).padStart(2, '0');
    const pngFilename = `${pYYYY}${pMM}${pDD}_${pHH}00Z.webp`;

    const githubFileUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${pngFilename}?ref=${dataBranch}`;
    console.log(`🔍 Prüfe GitHub-Repository auf existierendes WebP: ${pngFilename}`);

    const githubFileCheck = await fetch(githubFileUrl, { method: "HEAD", headers: githubHeaders });
    if (githubFileCheck.status === 200) {
      console.log(`💾 Datei ${pngFilename} existiert bereits im Repository. Lauf wird übersprungen.`);
      return;
    } else if (githubFileCheck.status !== 404) {
      console.log(`⚠️ Unerwarteter Status von GitHub API: ${githubFileCheck.status}. Abbruch.`);
      return;
    }

    // SCHRITT 4: GitHub Action triggern
    console.log(`🚀 Trigger Workflow ${workflowFileName} auf Branch '${pipelineBranch}' für Run ${targetRunString}Z...`);
    const dispatchUrl = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflowFileName}/dispatches`;

    const dispatchResponse = await fetch(dispatchUrl, {
      method: "POST",
      headers: githubHeaders,
      body: JSON.stringify({
        ref: pipelineBranch,
        inputs: {
          run: targetRunString
        }
      })
    });

    if (dispatchResponse.ok) {
      console.log("✅ GitHub Action erfolgreich gestartet.");
    } else {
      const errorText = await dispatchResponse.text();
      console.error(`❌ GitHub API Fehler (${dispatchResponse.status}):`, errorText);
    }   
  }
};