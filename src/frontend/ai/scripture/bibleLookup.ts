// AI AUTO SCRIPTURE - CHECKED BIBLE LOOKUPS
// json-bible never fails a lookup: a book number it does not have resolves to books[0] and a
// chapter it does not have resolves to chapters[0], with no error and no flag on the returned
// object beyond `index`. getChapter() then reports back the number that was REQUESTED rather
// than the one it landed on, so nothing downstream can tell the difference either.
//
// The failure that produced this file: a correctly detected Psalm 52:8 projected as "Genesis
// 52:8" carrying Genesis 1:8 text - the book fell back to books[0], the chapter fell back to
// chapters[0], and the reference label kept the detected numbers. Silently showing the wrong
// passage is the worst outcome this feature has on a live stage, so every AI passage lookup
// goes through these two and a miss skips the projection instead of guessing.

import type { BibleInstance, BookInstance, ChapterInstance } from "../../components/drawer/bible/scripture"

/** The requested book, or null when this bible does not have it (never a fallback book). */
export async function getBookExact(bible: BibleInstance, bookNumber: number | string): Promise<BookInstance | null> {
    const Book = await bible.getBook(bookNumber)
    return Book.index < 0 ? null : Book
}

/** The requested chapter, or null when that book does not have it (never a fallback chapter). */
export async function getChapterExact(Book: BookInstance, chapterNumber: number): Promise<ChapterInstance | null> {
    const Chapter = await Book.getChapter(chapterNumber)
    return Chapter.index < 0 ? null : Chapter
}

/** Whether a bible actually carries a book number - the 66 book canon assumption is only safe when it does. */
export function hasBookNumber(books: { number: number | string }[], bookNumber: number | string): boolean {
    return books.some((a) => Number(a.number) === Number(bookNumber))
}
