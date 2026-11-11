# Labelling facts: sensible or gibberish

You are labelling facts from a fact store that a program built by reading Wikipedia text on its own. The file `labels_packet.txt` has 300 numbered lines. Each line is one fact, written as:

`number. subject — relation — object`

For every line, answer **sensible** or **gibberish**.

**Sensible.** The line is a meaningful fact of the kind its relation names.
- The subject and the object are real names or terms.
- The object is the right kind of thing for the relation: a year for a birth year, a place for a birth place or a capital, a person for an author or a spouse, a language for a language, a kind of thing for "is a".
- Mark it sensible even if you are not sure it is true, or if the value may be a little off.

**Gibberish.** The line is broken or nonsense. For example:
- the object (or the subject) is a fragment of a sentence, leftover markup or code, or coordinates where a name belongs;
- the object is the wrong kind of thing for the relation (a profession as a nationality, a city as a nationality, a word that is not a place as a birth place);
- the object only repeats the subject or a piece of it, so it says nothing.

**The relations.**
- birth year, death year: a person's.
- founded: the year an organisation, place or work began.
- birth place, death place, capital, largest city, country, location, located in: places ("located in" is a larger place that holds the subject).
- is a: what kind of thing the subject is ("Loki — is a — god").
- occupation, nationality, genre, language, currency, party.
- family, order, class: the biological family, order or class of a living thing.
- author, director, composer, creator, founder, inventor, spouse, predecessor, successor: people.
- developer, publisher, owner: people, companies or bodies.

Do not look anything up. Answer from what you know and from the line itself.

**Output.** Exactly 300 lines, one per fact, in order: the number, a tab, and the word `sensible` or `gibberish`. No other text.
