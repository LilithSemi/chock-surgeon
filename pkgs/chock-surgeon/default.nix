{
  lib,
  buildPythonPackage,
  flakever,
  mkShell,
  python,
  pyprojectVersionPatchHook,
  setuptools,
}:
let
  versionParts = lib.splitString "-" flakever.version;
  pythonVersion =
    if builtins.length versionParts > 1 then
      "${builtins.head versionParts}+${lib.concatStringsSep "." (builtins.tail versionParts)}"
    else
      flakever.version;
in
buildPythonPackage (finalAttrs: {
  pname = "chock-surgeon";
  version = pythonVersion;
  pyproject = true;

  build-system = [ setuptools ];

  nativeBuildInputs = [ pyprojectVersionPatchHook ];

  src = lib.cleanSource ../../.;

  pythonImportsCheck = [ "chock_surgeon" ];

  checkPhase = ''
    runHook preCheck
    PYTHONDONTWRITEBYTECODE=1 ${python.interpreter} -m unittest discover -s tests -v
    runHook postCheck
  '';

  passthru.shell = mkShell {
    name = "chock-surgeon-dev-shell";

    packages = [
      python
    ];
  };
})
