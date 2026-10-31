{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flakever.url = "github:numinit/flakever";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
    verl-src = {
      url = "github:verl-project/verl/8df5baeed0f5ecdd7f870d83bc3d79194816e2a9";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      flakever,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      verl-src,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;

      nameValuePair = name: value: { inherit name value; };
      genAttrs = names: f: builtins.listToAttrs (map (n: nameValuePair n (f n)) names);
      allSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      flakeverConfig = flakever.lib.mkFlakever {
        inherit inputs;

        digits = [
          1
          2
          2
        ];
      };

      forAllSystems =
        f:
        genAttrs allSystems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ self.overlays.default ];
              config.allowUnfree = true;
            };
          }
        );

      treefmtEval = forAllSystems ({ pkgs, ... }: treefmt-nix.lib.evalModule pkgs (import ./treefmt.nix));

      verlWorkspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = verl-src; };

      mkVerlEnv =
        pkgs:
        let
          verlDependencies = {
            verl = [
              "vllm"
              "fsdp"
            ];
          };
          python = pkgs.python312;
          cudaWheelRoot = package: "${package}/${python.sitePackages}/nvidia/cu13";
          ncclWheelRoot = package: "${package}/${python.sitePackages}/nvidia/nccl";
          torchWheelRoot = package: "${package}/${python.sitePackages}/torch";
          exposeWheelLibraries =
            package: relativePath:
            package.overrideAttrs (old: {
              postInstall = (old.postInstall or "") + ''
                mkdir -p "$out/lib"
                for library in "$out/${relativePath}"/*.so*; do
                  ln -sf "$library" "$out/lib/$(basename "$library")"
                done
              '';
            });
          pythonBase = pkgs.callPackage pyproject-nix.build.packages { inherit python; };
          wheelOverrides = final: prev: {
            flash-attn = prev.flash-attn.overrideAttrs (old: {
              src = pkgs.fetchurl (
                if pkgs.stdenv.hostPlatform.isAarch64 then
                  {
                    url = "https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.13.0-cu130/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl";
                    hash = "sha256-e9eTdraBzKz0Sl1RojgWJXjQBlbSeeHSodaCAbrHYRI=";
                  }
                else
                  {
                    url = "https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.13.0-cu130/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl";
                    hash = "sha256-6mGnUCVLPwt5Fx1Y7j/kYn1gFBHumAHdFwWFXB7Oa/M=";
                  }
              );
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
                final.nvidia-cuda-runtime
              ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (torchWheelRoot final.torch)
                (cudaWheelRoot final.nvidia-cuda-runtime)
              ];
            });
            flashinfer-python = prev.flashinfer-python.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
                final.nvidia-cuda-runtime
              ];
            });
            humming-kernels = prev.humming-kernels.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.torch ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (torchWheelRoot final.torch)
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            vllm = prev.vllm.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
                final.nvidia-cuda-runtime
                final.nvidia-cuda-nvrtc
              ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (torchWheelRoot final.torch)
                (cudaWheelRoot final.nvidia-cuda-runtime)
                (cudaWheelRoot final.nvidia-cuda-nvrtc)
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            antlr4-python3-runtime = prev.antlr4-python3-runtime.overrideAttrs (old: {
              nativeBuildInputs =
                (old.nativeBuildInputs or [ ])
                ++ final.resolveBuildSystem {
                  setuptools = [ ];
                };
            });
            pylatexenc = prev.pylatexenc.overrideAttrs (old: {
              nativeBuildInputs =
                (old.nativeBuildInputs or [ ])
                ++ final.resolveBuildSystem {
                  setuptools = [ ];
                };
            });
            cuda-tile = prev.cuda-tile.overrideAttrs (_old: {
              src = pkgs.fetchurl (
                if pkgs.stdenv.hostPlatform.isAarch64 then
                  {
                    url = "https://pypi.nvidia.com/cuda-tile/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_aarch64.whl";
                    hash = "sha256-Li5f+GmSoPT0geF8ggwcE1Q4FzkJANUEHEletZlYDyE=";
                  }
                else
                  {
                    url = "https://pypi.nvidia.com/cuda-tile/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl";
                    hash = "sha256-t0wgNIIQ0hgs2Zig7LYMUYmJp5soWS1y64KUs43bk9c=";
                  }
              );
              nativeBuildInputs = [
                pythonBase.pyprojectWheelHook
                pkgs.autoPatchelfHook
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            nvidia-nvjitlink = exposeWheelLibraries prev.nvidia-nvjitlink "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cublas = exposeWheelLibraries prev.nvidia-cublas "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cuda-runtime = exposeWheelLibraries prev.nvidia-cuda-runtime "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cuda-nvrtc = exposeWheelLibraries prev.nvidia-cuda-nvrtc "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cuda-cupti = exposeWheelLibraries prev.nvidia-cuda-cupti "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-nccl-cu13 = exposeWheelLibraries prev.nvidia-nccl-cu13 "${python.sitePackages}/nvidia/nccl/lib";
            nvidia-curand = exposeWheelLibraries prev.nvidia-curand "${python.sitePackages}/nvidia/cu13/lib";
            apache-tvm-ffi = exposeWheelLibraries prev.apache-tvm-ffi "${python.sitePackages}/tvm_ffi/lib";
            z3-solver = exposeWheelLibraries prev.z3-solver "${python.sitePackages}/z3/lib";
            torch = exposeWheelLibraries (prev.torch.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.nvidia-cublas
                final.nvidia-cuda-cupti
                final.nvidia-cuda-nvrtc
                final.nvidia-cuda-runtime
                final.nvidia-cudnn-cu13
                final.nvidia-cufft
                final.nvidia-cufile
                final.nvidia-curand
                final.nvidia-cusolver
                final.nvidia-cusparse
                final.nvidia-cusparselt-cu13
                final.nvidia-nccl-cu13
                final.nvidia-nvshmem-cu13
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            })) "${python.sitePackages}/torch/lib";
            nvidia-cufile = exposeWheelLibraries (prev.nvidia-cufile.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.rdma-core ];
            })) "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cutlass-dsl-libs-base = prev.nvidia-cutlass-dsl-libs-base.overrideAttrs (_old: {
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            nvidia-nvshmem-cu13 = exposeWheelLibraries (prev.nvidia-nvshmem-cu13.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                pkgs.libfabric
                pkgs.openmpi
                pkgs.rdma-core
                pkgs.ucx
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            })) "${python.sitePackages}/nvidia/nvshmem/lib";
            nvidia-cusparselt-cu13 = exposeWheelLibraries prev.nvidia-cusparselt-cu13 "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cusparse = exposeWheelLibraries (prev.nvidia-cusparse.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.nvidia-nvjitlink ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (cudaWheelRoot final.nvidia-nvjitlink)
              ];
            })) "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cusolver = exposeWheelLibraries (prev.nvidia-cusolver.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.nvidia-cublas
                final.nvidia-cusparse
                final.nvidia-nvjitlink
              ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (cudaWheelRoot final.nvidia-cublas)
                (cudaWheelRoot final.nvidia-cusparse)
                (cudaWheelRoot final.nvidia-nvjitlink)
              ];
            })) "${python.sitePackages}/nvidia/cu13/lib";
            nvidia-cudnn-cu13 = exposeWheelLibraries (prev.nvidia-cudnn-cu13.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.nvidia-cublas ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (cudaWheelRoot final.nvidia-cublas)
              ];
            })) "${python.sitePackages}/nvidia/cudnn/lib";
            nvidia-cufft = exposeWheelLibraries (prev.nvidia-cufft.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.nvidia-nvjitlink ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (cudaWheelRoot final.nvidia-nvjitlink)
              ];
            })) "${python.sitePackages}/nvidia/cu13/lib";
            cupy-cuda13x = prev.cupy-cuda13x.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.nvidia-cublas
                final.nvidia-cuda-nvrtc
                final.nvidia-cufft
                final.nvidia-curand
                final.nvidia-cusolver
                final.nvidia-cusparse
                final.nvidia-nccl-cu13
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            nccl-extensions = prev.nccl-extensions.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.nvidia-nccl-cu13 ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (ncclWheelRoot final.nvidia-nccl-cu13)
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            nccl4py = prev.nccl4py.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.nvidia-nccl-cu13 ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (ncclWheelRoot final.nvidia-nccl-cu13)
              ];
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            pynvvideocodec = prev.pynvvideocodec.overrideAttrs (_old: {
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
            torchvision = prev.torchvision.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
                final.nvidia-cuda-runtime
              ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (torchWheelRoot final.torch)
                (cudaWheelRoot final.nvidia-cuda-runtime)
              ];
            });
            torchaudio = prev.torchaudio.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.torch ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (torchWheelRoot final.torch)
              ];
            });
            torchcodec = prev.torchcodec.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.torch
                pkgs.ffmpeg-headless
                pkgs.libheif
              ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (torchWheelRoot final.torch)
              ];
              autoPatchelfIgnoreMissingDeps = [
                "libavcodec.so.58"
                "libavcodec.so.59"
                "libavcodec.so.60"
                "libavcodec.so.61"
                "libavcodec.so.62"
                "libavdevice.so.58"
                "libavdevice.so.59"
                "libavdevice.so.60"
                "libavdevice.so.61"
                "libavdevice.so.62"
                "libavfilter.so.7"
                "libavfilter.so.8"
                "libavfilter.so.9"
                "libavfilter.so.10"
                "libavfilter.so.11"
                "libavformat.so.58"
                "libavformat.so.59"
                "libavformat.so.60"
                "libavformat.so.61"
                "libavformat.so.62"
                "libavutil.so.56"
                "libavutil.so.57"
                "libavutil.so.58"
                "libavutil.so.59"
                "libavutil.so.60"
                "libswresample.so.3"
                "libswresample.so.4"
                "libswresample.so.5"
                "libswresample.so.6"
                "libswscale.so.5"
                "libswscale.so.6"
                "libswscale.so.7"
                "libswscale.so.8"
                "libswscale.so.9"
              ];
            });
            torch-c-dlpack-ext = prev.torch-c-dlpack-ext.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.torch ];
              runtimeDependencies = (old.runtimeDependencies or [ ]) ++ [
                (torchWheelRoot final.torch)
              ];
            });
            xgrammar = prev.xgrammar.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.apache-tvm-ffi ];
            });
            tilelang = prev.tilelang.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.apache-tvm-ffi
                final.z3-solver
              ];
            });
          };
          pythonSet = pythonBase.overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              (verlWorkspace.mkPyprojectOverlay {
                sourcePreference = "wheel";
                dependencies = verlDependencies;
              })
              wheelOverrides
            ]
          );
        in
        (pythonSet.mkVirtualEnv "chock-surgeon-verl-env" verlDependencies).overrideAttrs (_old: {
          venvSkip = [ "lib/python${python.pythonVersion}/site-packages/build_backend.py" ];
        });
    in
    {
      versionTemplate = "0.1.0-<lastModifiedDate>-<rev>";

      overlays.default = final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (pyfinal: pyprev: {
            chock-surgeon = pyfinal.callPackage ./pkgs/chock-surgeon {
              flakever = flakeverConfig;
            };
          })
        ];

        inherit (final.python3Packages) chock-surgeon;
      };

      devShells = forAllSystems (
        { pkgs, ... }:
        let
          verlEnv = mkVerlEnv pkgs;
        in
        {
          default = pkgs.chock-surgeon.shell;
        }
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          train = pkgs.mkShell {
            name = "chock-surgeon-verl";
            packages = [
              verlEnv
              pkgs.chock-surgeon
              pkgs.uv
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = "${verlEnv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
              LD_LIBRARY_PATH = "${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}:/run/opengl-driver/lib";
            };
            shellHook = ''
              unset PYTHONPATH
              export CHOCK_SURGEON_ROOT=$(git rev-parse --show-toplevel)
              export PYTHONPATH="$CHOCK_SURGEON_ROOT/src"
            '';
          };
        }
      );

      packages = forAllSystems (
        { system, pkgs, ... }:
        {
          default = pkgs.chock-surgeon;
        }
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          verl-env = mkVerlEnv pkgs;
        }
      );

      formatter = forAllSystems ({ system, ... }: treefmtEval.${system}.config.build.wrapper);

      checks = forAllSystems (
        { system, pkgs, ... }:
        {
          inherit (pkgs) chock-surgeon;
          formatting = treefmtEval.${system}.config.build.check self;
        }
      );
    };
}
