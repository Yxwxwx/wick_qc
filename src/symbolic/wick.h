#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace wickqc::symbolic {

enum class OrbitalSpace : std::uint8_t {
  kGeneral = 0,
  kInactive = 1,
  kActive = 2,
  kSingle = 4,
  kExternal = 8,
};

enum class Spin : std::uint8_t {
  kNone = 0,
  kAlpha = 1,
  kBeta = 2,
};

struct IndexDomain {
  std::uint8_t orbital_spaces = 0;
  std::uint8_t spins = 0;

  [[nodiscard]] std::strong_ordering operator<=>(const IndexDomain& other) const noexcept;
  bool operator==(const IndexDomain&) const = default;
  [[nodiscard]] bool IsConcrete() const noexcept;
  [[nodiscard]] bool IsCompatibleWith(const IndexDomain& other) const noexcept;
};

struct Index {
  std::string name;
  IndexDomain domain;

  [[nodiscard]] std::strong_ordering operator<=>(const Index& other) const;
  bool operator==(const Index&) const = default;
  [[nodiscard]] bool HasTypes() const noexcept;
  [[nodiscard]] bool IsShort() const noexcept;
  [[nodiscard]] Index Untyped() const;
  [[nodiscard]] std::size_t Hash() const noexcept;
  void Save(std::ostream& output) const;
  [[nodiscard]] static Index Load(std::istream& input);
};

class IndexRegistry {
 public:
  void Add(OrbitalSpace space, std::string_view names, Spin spin = Spin::kNone);
  [[nodiscard]] Index Resolve(std::string_view name) const;
  [[nodiscard]] std::vector<Index> Parse(std::string_view names) const;
  [[nodiscard]] std::set<Index> ParseSet(std::string_view names) const;
  [[nodiscard]] std::vector<IndexDomain> ConcreteDomains(std::string_view name) const;

 private:
  std::map<std::string, IndexDomain> domains_;
};

struct SignedPermutation {
  std::vector<std::size_t> order;
  int sign = 1;

  [[nodiscard]] std::strong_ordering operator<=>(const SignedPermutation& other) const;
  bool operator==(const SignedPermutation&) const = default;
  [[nodiscard]] static SignedPermutation Identity(std::size_t rank);
  [[nodiscard]] SignedPermutation Compose(const SignedPermutation& other) const;
  [[nodiscard]] std::size_t Hash() const noexcept;
  void Save(std::ostream& output) const;
  [[nodiscard]] static SignedPermutation Load(std::istream& input);
};

class TensorSymmetry {
 public:
  TensorSymmetry() = default;
  TensorSymmetry(std::size_t rank, const std::vector<SignedPermutation>& generators);

  [[nodiscard]] static TensorSymmetry None(std::size_t rank);
  [[nodiscard]] static TensorSymmetry TwoSymmetric();
  [[nodiscard]] static TensorSymmetry TwoAntisymmetric();
  // Antisymmetry under interchange of the two index pairs (CT amplitudes).
  [[nodiscard]] static TensorSymmetry CanonicalTransformation();
  [[nodiscard]] static TensorSymmetry FourAntisymmetric();
  [[nodiscard]] static TensorSymmetry QuantumChemistryChemists();
  [[nodiscard]] static TensorSymmetry QuantumChemistryPhysicists();
  [[nodiscard]] static TensorSymmetry SpinFree(std::size_t order, bool hermitian = false);
  // Independent antisymmetry within each half; no Hermitian pair exchange.
  [[nodiscard]] static TensorSymmetry PairAntisymmetric(std::size_t order);
  [[nodiscard]] static TensorSymmetry All(std::size_t rank);
  // Preserve an already enumerated group, including the supplied element order.
  [[nodiscard]] static TensorSymmetry FromElements(std::vector<SignedPermutation> elements);

  [[nodiscard]] const std::vector<SignedPermutation>& Elements() const;

 private:
  std::vector<SignedPermutation> elements_;
};

class SymmetryRegistry {
 public:
  void Add(std::string name, std::size_t rank, TensorSymmetry symmetry);
  [[nodiscard]] TensorSymmetry Lookup(std::string_view name, std::size_t rank) const;

 private:
  std::map<std::pair<std::string, std::size_t>, TensorSymmetry> symmetries_;
};

enum class TensorKind : std::uint8_t {
  kGeneric,
  kCreation,
  kAnnihilation,
  kSpinFree,
  kDelta,
};

struct Tensor {
  std::string name;
  std::vector<Index> indices;
  TensorKind kind = TensorKind::kGeneric;
  // Omitted symmetry means no extra symmetry, not an empty allowed group.
  TensorSymmetry symmetry = TensorSymmetry::None(indices.size());

  [[nodiscard]] static Tensor Parse(std::string_view text,
                                    const IndexRegistry& indices,
                                    const SymmetryRegistry& symmetries);
  [[nodiscard]] bool IsFermionOperator() const noexcept;
  [[nodiscard]] bool operator==(const Tensor& other) const;
  [[nodiscard]] bool operator<(const Tensor& other) const;
  [[nodiscard]] int CompareFermiClass(const Tensor& rhs) const;
  [[nodiscard]] Tensor Permute(const SignedPermutation& permutation) const;
  [[nodiscard]] Tensor Canonicalize(double& coefficient) const;
  [[nodiscard]] Tensor RestrictSymmetry() const;
  [[nodiscard]] std::map<std::string, std::string> IndexMapTo(const Tensor& other) const;
  [[nodiscard]] std::vector<std::map<std::string, std::string>> IndexPermutations() const;
  [[nodiscard]] std::string ToString(const SignedPermutation& permutation) const;
  [[nodiscard]] std::string PermutationRules() const;
  [[nodiscard]] int SpinTag() const;
  void SetSpinTag(int tag);
  void Save(std::ostream& output) const;
  [[nodiscard]] static Tensor Load(std::istream& input);
};

struct Term {
  double coefficient = 1.0;
  std::vector<Tensor> tensors;
  std::vector<Index> summed_indices;
  [[nodiscard]] bool operator==(const Term& other) const;
  [[nodiscard]] bool operator<(const Term& other) const;
  [[nodiscard]] bool SameForm(const Term& other) const;
  [[nodiscard]] std::set<Index> UsedIndices() const;
  [[nodiscard]] std::set<std::string> UsedIndexNames() const;
  [[nodiscard]] std::set<std::string> SummedIndexNames() const;
  [[nodiscard]] std::map<int, int> SpinTagCounts() const;
  [[nodiscard]] bool HasOperatorsIn(OrbitalSpace space) const;
  [[nodiscard]] Term Canonicalize() const;
  // Canonicalize each tensor and sort commuting factors; preserve operator order.
  [[nodiscard]] Term SortFactors() const;
  void Save(std::ostream& output) const;
  [[nodiscard]] static Term Load(std::istream& input);
};

class Expression {
 public:
  Expression() = default;
  explicit Expression(Term term);
  explicit Expression(std::vector<Term> terms);

  [[nodiscard]] static Expression Parse(std::string_view text,
                                        const IndexRegistry& indices,
                                        const SymmetryRegistry& symmetries);
  [[nodiscard]] static std::pair<Tensor, Expression> ParseDefinition(std::string_view text,
                                                                     const IndexRegistry& indices,
                                                                     const SymmetryRegistry& symmetries);

  // Orbital alternatives of bound indices are enumerated; spin masks are kept.
  [[nodiscard]] Expression SplitIndexDomains() const;
  // Normal ordering alone, without the domain splitting performed by Expand.
  [[nodiscard]] Expression NormalOrder(int max_uncontracted = -1,
                                       bool skip_contractions = false,
                                       bool compact_spin_free = true) const;

  // Compact spin-free output omits residual paired inactive operators.
  // Disable it to retain explicit operators with summed spin tags.
  [[nodiscard]] Expression Expand(int max_uncontracted = -1,
                                  bool skip_contractions = false,
                                  bool compact_spin_free = true) const;
  [[nodiscard]] Expression Simplify(double tolerance = 1.0e-12) const;
  [[nodiscard]] Expression SortFactors() const;
  [[nodiscard]] Expression SimplifyDeltas() const;
  [[nodiscard]] Expression RemoveZeros(double tolerance = 1.0e-12) const;
  [[nodiscard]] Expression MergeTerms(double tolerance = 1.0e-12) const;
  [[nodiscard]] Expression Conjugate() const;
  // Simultaneous renaming, including bound summation indices.
  [[nodiscard]] Expression RenameIndices(const std::map<std::string, std::string>& names) const;
  // Substitute each matching tensor once, with fresh dummy indices per use.
  [[nodiscard]] Expression Substitute(const std::map<std::string, std::pair<Tensor, Expression>>& definitions) const;
  [[nodiscard]] Expression RemoveExternal() const;
  [[nodiscard]] Expression RemoveInactive() const;
  [[nodiscard]] Expression AddSpinFreeTransposeSymmetry() const;

  [[nodiscard]] const std::vector<Term>& Terms() const noexcept;
  [[nodiscard]] bool operator==(const Expression& other) const;
  [[nodiscard]] bool operator<(const Expression& other) const;
  [[nodiscard]] bool Empty() const noexcept;
  // Native-ABI binary layout interoperable with the supplied Wick reference.
  void Save(std::ostream& output) const;
  [[nodiscard]] static Expression Load(std::istream& input);

  friend Expression operator+(const Expression& lhs, const Expression& rhs);
  friend Expression operator-(const Expression& lhs, const Expression& rhs);
  friend Expression operator*(const Expression& lhs, const Expression& rhs);
  friend Expression operator*(double scalar, const Expression& expression);
  friend Expression operator*(const Expression& expression, double scalar);
  friend Expression Commutator(const Expression& lhs, const Expression& rhs);
  // Multiply with fresh dummy labels, then bind every orbital index.
  friend Expression FullySummedProduct(const Expression& lhs, const Expression& rhs);
  friend std::ostream& operator<<(std::ostream& output, const Expression& expression);

 private:
  std::vector<Term> terms_;
};

} // namespace wickqc::symbolic
