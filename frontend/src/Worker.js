/* eslint-disable */
// Otherwise webpack fails silently
// https://github.com/facebook/create-react-app/issues/8014

import * as Comlink from 'comlink';
import pythonCoreUrl from "./python_core.tar.load_by_url"
import {
  loadPyodideAndPackage,
  makeRunnerCallback,
  pyodideExpose,
  PyodideFatalErrorReloader
} from "pyodide-worker-runner";

const reloader = new PyodideFatalErrorReloader(async () => {
  const pyodide = await loadPyodideAndPackage({url: pythonCoreUrl, format: "tar"});
  pyodide.pyimport("core.init_pyodide").init(process.env.REACT_APP_LANGUAGE);
  return pyodide;
});

let programCount = 1;

const runCode = pyodideExpose(
  async function (extras, entry, outputCallback, inputCallback) {
    let outputPromise;
    const callback = makeRunnerCallback(extras, {
      input: () => inputCallback(),
      output: (parts) => {
        outputPromise = outputCallback(parts);
      },
    });

    return await reloader.withPyodide(async (pyodide) => {
      const pyodide_worker_runner = pyodide.pyimport("pyodide_worker_runner");
      try {
        await pyodide_worker_runner.install_imports(entry.input);
      } catch (e) {
        console.error(e);
      }

      const checkerModule = pyodide.pyimport("core.checker");
      checkerModule.default_runner.set_filename(`/my_program_${programCount++}.py`)
      const result = checkerModule.check_entry(entry, callback);
      await outputPromise;
      return result.toJs({dict_converter: Object.fromEntries});
    });
  },
);

Comlink.expose({runCode});
