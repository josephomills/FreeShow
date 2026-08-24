// AI AUTO SCRIPTURE - spoken names for installed translations
//
// Bibles are stored under their short names ("GNT"), but a preacher names them in full: "First
// Samuel ten and five, Good News". The spoken name reaches every consumer of the translation
// table (voice commands and reference detection) only if it is IN the table, and nothing about
// an installed bible carries it - so the identities live here. They are universal facts about
// the translations, not user data, which is why a static table is the right shape.

const SPOKEN_TRANSLATION_NAMES: Record<string, string[]> = {
    AMP: ["Amplified", "Amplified Bible"],
    AMPC: ["Amplified Classic"],
    ASV: ["American Standard", "American Standard Version"],
    BSB: ["Berean", "Berean Standard"],
    CEB: ["Common English"],
    CEV: ["Contemporary English"],
    CSB: ["Christian Standard"],
    ERV: ["Easy to Read"],
    ESV: ["English Standard", "English Standard Version"],
    GNT: ["Good News", "Good News Translation", "Good News Bible"],
    GW: ["God's Word"],
    HCSB: ["Holman"],
    KJV: ["King James", "King James Version", "Authorized Version"],
    LEB: ["Lexham"],
    MEV: ["Modern English"],
    MSG: ["The Message"],
    NASB: ["New American Standard", "New American Standard Bible"],
    NCV: ["New Century"],
    NET: ["New English Translation"],
    NIRV: ["New International Reader's Version"],
    NIV: ["New International", "New International Version"],
    NKJV: ["New King James", "New King James Version"],
    NLT: ["New Living", "New Living Translation"],
    NOG: ["Names of God"],
    NRSV: ["New Revised Standard"],
    RSV: ["Revised Standard", "Revised Standard Version"],
    TLB: ["The Living Bible"],
    TLV: ["Tree of Life"],
    TPT: ["The Passion", "Passion Translation"],
    WEB: ["World English"],
    YLT: ["Young's Literal"]
}

/** The spoken full names for a stored bible name, or none when it is not a known short name. */
export function spokenTranslationNames(storedName: string): string[] {
    return SPOKEN_TRANSLATION_NAMES[storedName.trim().toUpperCase()] || []
}
