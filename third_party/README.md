# third_party/

This directory contains external dependencies used by the project.

Currently included:

---

## Eigen (as Git submodule)

Eigen is used **only for validation tests** (`tests/test_lu_factorization_*.cpp`).
It is *not* bundled into the repository directly.  
Instead, we track a specific upstream commit via a Git submodule.

### Fetching Eigen

If you cloned the repository normally:

```bash
git submodule update --init --recursive
```

Or clone everything in one step:

```bash
git clone --recursive <url-to-this-repo>
```

After initialization, Eigen's headers are available under **third_party/eigen**.

### Updating the Eigen version

To update Eigen:

```bash
cd third_party/eigen
git fetch origin
git checkout <tag-or-commit>
cd ../..
git add third_party/eigen
git commit -m "Update Eigen submodule to version <tag>"
```