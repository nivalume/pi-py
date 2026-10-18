// Usage: node interop.mjs write|read sqlite|jsonl <store path> <fixture dir> [<results file>] (see generate.py).
import { readFileSync, writeFileSync } from "node:fs";
import { BACKGROUND_CONTEXT as C } from "@earendil-works/chord/context";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";
import { openNodeJsonlStorage } from "@earendil-works/pi-durable/storage/jsonl/node";

const [mode, backend, path, dataDir, outFile] = process.argv.slice(2);
const open = () => (backend === "sqlite" ? openNodeSqliteStorage(path) : openNodeJsonlStorage(path, C));
const storage = await open();
if (mode === "write") {
	const commits = JSON.parse(readFileSync(`${dataDir}/commits.json`, "utf8"));
	for (const writes of commits) await storage.commit(writes, C);
} else {
	const reads = JSON.parse(readFileSync(`${dataDir}/reads.json`, "utf8"));
	const results = [];
	for (const [method, ...args] of reads) {
		try {
			let value;
			switch (method) {
				case "conversation": value = await storage.conversation(args[0], C); break;
				case "scanConversations": value = await storage.scanConversations(args[0], args[1], undefined, C); break;
				case "entry": value = await storage.entry(args[0], C); break;
				case "visibleEntry": value = await storage.entry(args[0], args[1], C); break;
				case "findLatestHeadMarker": value = await storage.findLatestHeadMarker(args[0], args[1] ?? undefined, C); break;
				case "scanEntries": value = await storage.scanEntries(args[0], args[1], undefined, C); break;
				case "task": value = await storage.task(args[0], C); break;
				case "scanTasks": value = await storage.scanTasks(args[0], args[1], undefined, C); break;
				case "submission": value = await storage.submission(args[0], C); break;
				case "scanSubmissions": value = await storage.scanSubmissions(args[0], args[1], undefined, C); break;
				case "submissionByRequest": value = await storage.submissionByRequest(args[0], args[1], C); break;
				case "document": value = await storage.document(args[0], args[1], C); break;
				case "findDocument": value = await storage.findDocument(args[0], args[1], C); break;
				case "scanDocuments": value = await storage.scanDocuments(args[0], args[1], undefined, C); break;
				case "mintId": value = await storage.mintId(); break;
				case "commit": value = await storage.commit(args[0], C); break;
				default: throw new Error(`unknown ${method}`);
			}
			results.push({ ok: value === undefined ? null : value });
		} catch (error) {
			results.push({ error: String(error.message) });
		}
	}
	writeFileSync(outFile, JSON.stringify(results, null, 1));
}
await storage.close(C);
