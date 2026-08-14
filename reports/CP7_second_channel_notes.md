# CP-7: a second constraint family on real crystals, and the limit of the expressible class

`results/CP7_second_channel.json`

Two results. Charge neutrality, the plan's family (i), is outside the class this layer can
express, and the reason is structural rather than an implementation gap. A different second
family — valence electron density — is expressible, is not a restatement of density, and
solves jointly with density on real crystal prefixes at solver precision.

## Charge neutrality is not a linear functional

The layer constrains a linear functional of the next token, sum_i a_i q_i = b. Charge
neutrality as SMACT defines it, and as this project's own plausibility screen measures it
at a real-data rate of 0.7150, is an existential predicate: a composition is balanced if
there exists an assignment of formal charges, drawn from each element's allowed oxidation
states, summing to zero. The quantifier is doing nearly all the work. Of 87 elements in
the MP-20 training split, 85 have tabulated oxidation states and 71 of those are
multivalent, so 81.5% of element-occurrences involve an element whose charge is not
determined by its identity.

Collapsing the predicate to one charge per element makes it linear and destroys it.
Measured on 4,000 training structures:

| quantity | value |
|---|---|
| SMACT calls balanced | 71.08% |
| linear proxy (first-listed oxidation state) calls balanced | 7.95% |
| the two agree | 36.78% |
| proxy recall of the SMACT-balanced set | 11.12% |
| false negatives | 2,527 |
| false positives | 2 |

A Regime E target built on that proxy would steer a quantity with almost nothing to do
with the screen the paper reports. So charge neutrality is recorded as outside the
expressible class. That is a scope result worth stating plainly rather than a gap to be
closed, and it sharpens the method's boundary: the layer expresses additive functionals of
the emitted symbols, not predicates requiring a search over latent assignments.

## The second family: valence electron density

Valence electron count summed over the formula, divided by cell volume. It is additive in
element counts, so it fits the channel construction exactly; it is defined for every
element in the split (87 elements, against two elements missing a Pauling
electronegativity had that been chosen instead); it is the quantity electron-count rules
are written in; and it is not directly emitted — the model emits elements, counts and
lattice parameters, never an electron count. That last point answers the standing
criticism of the density result, which is that density is close to something the model
literally writes down.

| statistic | mass density (g/cm^3) | valence electron density (e/A^3) |
|---|---|---|
| mean | 6.679 | 0.4839 |
| sd | 2.942 | 0.2019 |
| min | 0.507 | 0.0092 |
| max | 21.976 | 1.5713 |

Correlation across the 26,910 training structures is 0.725. That number is
load-bearing in both directions. Too low and the pair would be nearly orthogonal, making
composition unrealistically easy; too high and the two would be one constraint written
twice, whose multiplier Hessian is singular by construction. 0.725 is high enough to
be a real test and low enough to stay conditioned, and a test asserts it inside that band
rather than merely recording it.

The mass channel was generalized over its weight table rather than copied, since the
prefix logic is the subtle part — an element token commits nothing until its count
arrives. `weight_channel(ids, V, ATOMIC_MASS)` is verified bit-identical to
`mass_channel` at every prefix length, so no published density artifact changes.

## Joint steering works, and only at element positions

Both constraints solved jointly on 64 real crystal prefixes: all converged, worst
residual 7.67e-13, simplex error 4.4e-16, zero mass on grammatically illegal
tokens, and no ill-conditioned case. Hessian condition number median 124.2, 95th
percentile 178.9.

That is conditional on where the constraint acts, which is the substantive finding. Of the
positions offering a grammatical choice with channel variation, 260 carry two
independent constraints and 196 carry only one. The split is not statistical: at a
count position with pending element E the tokens are `cnt:k`, so mass contributes
MASS[E]·k and valence contributes VALENCE[E]·k, both proportional to k. The pair is
exactly collinear there — measured singular-value ratios of order 1e-10 against 5e-2 at
element positions — so joint steering of two additive-per-element quantities is possible
only at element positions, where the two weight tables differ across elements. Both facts
are pinned by test.

This was found the hard way. An earlier run reported six of sixteen prefixes failing to
converge with a median condition number of 105, which is self-contradictory. The cause was
a one-position mask misalignment: `masks_for` returns the mask at each existing position,
so index −1 belongs to the token already emitted. The channel was evaluated for the next
token while the mask constrained the previous one, yielding a support on which the mass
channel was identically zero — an attainable range of a single point, which surfaced as
ill-conditioning rather than as a wrong position. The sampling loop documents the same trap
and solves it by padding one column; this code now does the same, and a test asserts the
channel is nonzero on the next-token support so the misalignment cannot return silently.

## Priority 3 is descoped, deliberately

The plan asks for a held-out prediction of a serialization's steering range. Three
serializations exist and only one has a measured frontier, so fitting on two and testing
on one is a residual rather than a held-out prediction. Buying the frontier on two more
serializations, roughly 17.6 GPU-hours, would raise that to three fit points and one test
point, which is the same objection at higher cost.

Priority 3 is therefore recorded as a correlational claim across the serializations that
exist, stated as such, with the predictive version deferred until five or six
serializations can be budgeted together. This is a scope decision, not a pending task, and
it should not be silently re-scoped into a prediction later.
