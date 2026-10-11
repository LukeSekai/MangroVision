"""Organization-based usernames and memorable, independently generated passwords."""

import re
import secrets
import unicodedata


WORDS = tuple('''
acorn amber apple apron arch arrow autumn bamboo basket beach beacon berry
bird bloom blossom boat breeze bridge brook bucket butter cabin cactus candle
canyon cedar cherry chestnut circle cliff cloud clover coast cocoa coconut coral
cove crane creek crown dawn delta dune eagle earth elm ember evening fern field
finch fire flame flower forest fountain frost garden gate ginger glade glass globe
gold grain grape grass grove harbor hazel heart hedge heron hill honey island ivy
jade jasmine jewel juniper kite lake lantern lark laurel leaf lemon lily linen lion
lotus maple marble meadow melon mint moon moss mountain nest oak ocean olive
orchard orchid otter palm panda pearl pebble peach pear petal pine planet plum
pond poppy prairie pumpkin quartz rabbit rain rainbow reed reef ribbon river
robin rock rose sail sand seed shell shore sky snow sparrow spring star stone
stream sun sunrise sunset swan thyme tide timber trail tree tulip valley velvet
violet walnut wave wheat willow wind wing winter wood wren
'''.split())


def organization_slug(name, organization_id=None):
    ascii_name = unicodedata.normalize('NFKD', str(name or '')).encode('ascii', 'ignore').decode()
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_name.lower()).strip('-')[:48].rstrip('-')
    if slug:
        return slug
    return f'organization-{organization_id}' if organization_id is not None else 'organization'


def organization_password(name):
    """Name prefix + four readable words + eight digits (>54 bits of randomness)."""
    prefix = ''.join(part.capitalize() for part in organization_slug(name).split('-')[:2])[:18]
    words = [secrets.choice(WORDS).capitalize() for _ in range(4)]
    return '-'.join([prefix, *words, f'{secrets.randbelow(100_000_000):08d}'])


def available_username(conn, name, organization_id):
    base = organization_slug(name, organization_id)
    # Serialize automatic accounts with the same readable slug. Manual accounts
    # remain protected by the database's unique username constraint.
    conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(?, 0))', (f'organization-login:{base}',))
    candidate = base
    suffix = 2
    while conn.execute('SELECT id FROM planters WHERE lower(username) = lower(?)', (candidate,)).fetchone():
        candidate = f'{base}-{suffix}'
        suffix += 1
    return candidate
