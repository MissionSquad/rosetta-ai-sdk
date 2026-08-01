import {
  LeadingThinkingTagParser,
  cloneOpenAICompatibleAssistantProviderState,
  cloneReplayRecordArray,
  extractOpenAIThinking,
  isUnknownRecord,
  parseLeadingThinkingTags
} from '../../../../src/core/mapping/openai-thinking'

interface CollectedParseResult {
  thinking: string
  content: string
  signalEvents: number
  hasThinkingSignal: boolean
}

function parseChunks(chunks: string[]): CollectedParseResult {
  const parser = new LeadingThinkingTagParser()
  let thinking = ''
  let content = ''
  let signalEvents = 0

  for (const chunk of chunks) {
    const result = parser.push(chunk)
    thinking += result.thinkingDelta
    content += result.contentDelta
    if (result.hasThinkingSignal) signalEvents += 1
  }

  const finalResult = parser.finish()
  thinking += finalResult.thinkingDelta
  content += finalResult.contentDelta
  if (finalResult.hasThinkingSignal) signalEvents += 1

  return { thinking, content, signalEvents, hasThinkingSignal: parser.hasThinkingSignal }
}

describe('OpenAI-compatible thinking extraction', () => {
  describe('runtime guards and replay cloning', () => {
    it('recognizes non-array records only', () => {
      expect(isUnknownRecord({ key: 'value' })).toBe(true)
      expect(isUnknownRecord(Object.create(null))).toBe(true)
      expect(isUnknownRecord(null)).toBe(false)
      expect(isUnknownRecord([])).toBe(false)
      expect(isUnknownRecord('record')).toBe(false)
      expect(isUnknownRecord(1)).toBe(false)
    })

    it('deep-clones arrays of records without retaining nested array or record references', () => {
      const source = [
        {
          type: 'reasoning.text',
          text: 'Inspecting',
          metadata: { nested: [{ value: 'original' }] }
        }
      ]

      const cloned = cloneReplayRecordArray(source)

      expect(cloned).toEqual(source)
      expect(cloned).not.toBe(source)
      expect(cloned?.[0]).not.toBe(source[0])
      source[0].metadata.nested[0]!.value = 'mutated'
      expect(cloned).toEqual([
        {
          type: 'reasoning.text',
          text: 'Inspecting',
          metadata: { nested: [{ value: 'original' }] }
        }
      ])
    })

    it.each([null, undefined, 'not-an-array', {}, [null], [{ type: 'text' }, 'invalid']])(
      'rejects a replay value that is not entirely an array of records: %p',
      value => {
        expect(cloneReplayRecordArray(value)).toBeUndefined()
      }
    )

    it('accepts an empty replay array as a valid cloned array', () => {
      const source: Record<string, unknown>[] = []
      const cloned = cloneReplayRecordArray(source)
      expect(cloned).toEqual([])
      expect(cloned).not.toBe(source)
    })

    it('clones both supported assistant provider-state arrays and drops unrelated keys', () => {
      const source = {
        reasoningDetails: [{ type: 'reasoning.encrypted', data: 'ciphertext' }],
        structuredContent: [{ type: 'text', text: 'Answer' }],
        ignored: [{ secret: 'not-part-of-contract' }]
      }

      const cloned = cloneOpenAICompatibleAssistantProviderState(source)

      expect(cloned).toEqual({
        reasoningDetails: [{ type: 'reasoning.encrypted', data: 'ciphertext' }],
        structuredContent: [{ type: 'text', text: 'Answer' }]
      })
      expect(cloned?.reasoningDetails).not.toBe(source.reasoningDetails)
      expect(cloned?.structuredContent).not.toBe(source.structuredContent)
    })

    it('retains each independently valid provider-state array and rejects invalid fields', () => {
      expect(
        cloneOpenAICompatibleAssistantProviderState({
          reasoningDetails: [{ type: 'reasoning.summary', summary: 'Summary' }],
          structuredContent: ['invalid']
        })
      ).toEqual({ reasoningDetails: [{ type: 'reasoning.summary', summary: 'Summary' }] })

      expect(
        cloneOpenAICompatibleAssistantProviderState({
          reasoningDetails: 'invalid',
          structuredContent: [{ type: 'text', text: 'Answer' }]
        })
      ).toEqual({ structuredContent: [{ type: 'text', text: 'Answer' }] })
    })

    it.each([null, [], 'invalid', {}, { reasoning_details: [] }, { reasoningDetails: [null] }])(
      'rejects invalid or empty assistant provider state: %p',
      value => {
        expect(cloneOpenAICompatibleAssistantProviderState(value)).toBeUndefined()
      }
    )
  })

  describe('provider field precedence', () => {
    it('concatenates textual reasoning_details entries in array order and prefers them over every alias', () => {
      const reasoningDetails = [
        { type: 'reasoning.text', text: 'Text A. ' },
        { type: 'reasoning.summary', summary: 'Summary B. ' },
        { type: 'reasoning.text', text: 'Text C.' }
      ]

      const result = extractOpenAIThinking({
        reasoning_details: reasoningDetails,
        reasoning_content: 'DeepSeek duplicate',
        reasoning: 'OpenRouter duplicate',
        thinking: 'Ollama duplicate',
        analysis: 'Analysis duplicate',
        content: [
          { type: 'thinking', thinking: [{ type: 'text', text: 'Mistral duplicate' }] },
          { type: 'text', text: 'Answer A.' },
          { type: 'text', text: ' Answer B.' }
        ]
      })

      expect(result).toEqual({
        thinkingDeltas: ['Text A. Summary B. Text C.'],
        answerDeltas: ['Answer A.', ' Answer B.'],
        hasThinkingSignal: true,
        reasoningDetails,
        structuredContent: [
          { type: 'thinking', thinking: [{ type: 'text', text: 'Mistral duplicate' }] },
          { type: 'text', text: 'Answer A.' },
          { type: 'text', text: ' Answer B.' }
        ]
      })
      expect(result.reasoningDetails).not.toBe(reasoningDetails)
    })

    it('uses the first non-empty string alias in the required precedence order', () => {
      const cases: Array<{ input: Record<string, unknown>; expected: string }> = [
        {
          input: {
            reasoning_content: 'DeepSeek/Qwen/xAI',
            reasoning: 'Groq/OpenRouter',
            thinking: 'Ollama',
            analysis: 'Fallback'
          },
          expected: 'DeepSeek/Qwen/xAI'
        },
        {
          input: { reasoning_content: '', reasoning: 'Groq/OpenRouter', thinking: 'Ollama', analysis: 'Fallback' },
          expected: 'Groq/OpenRouter'
        },
        {
          input: { reasoning_content: '', reasoning: '', thinking: 'Ollama', analysis: 'Fallback' },
          expected: 'Ollama'
        },
        { input: { reasoning_content: '', reasoning: '', thinking: '', analysis: 'Fallback' }, expected: 'Fallback' }
      ]

      for (const { input, expected } of cases) {
        expect(extractOpenAIThinking(input)).toEqual({
          thinkingDeltas: [expected],
          answerDeltas: [],
          hasThinkingSignal: true
        })
      }
    })

    it('treats a whitespace-only alias as non-empty disclosed text', () => {
      expect(extractOpenAIThinking({ reasoning_content: '  ', reasoning: 'later alias' })).toEqual({
        thinkingDeltas: ['  '],
        answerDeltas: [],
        hasThinkingSignal: true
      })
    })

    it('falls through encrypted-only details to a textual alias while retaining the encrypted signal and replay', () => {
      expect(
        extractOpenAIThinking({
          reasoning_details: [{ type: 'reasoning.encrypted', data: 'never-render-this' }],
          reasoning_content: 'Disclosed reasoning',
          content: 'Answer'
        })
      ).toEqual({
        thinkingDeltas: ['Disclosed reasoning'],
        answerDeltas: ['Answer'],
        hasThinkingSignal: true,
        reasoningDetails: [{ type: 'reasoning.encrypted', data: 'never-render-this' }]
      })
    })

    it('surfaces an encrypted-only signal without rendering encrypted bytes', () => {
      expect(
        extractOpenAIThinking({
          reasoning_details: [{ type: 'reasoning.encrypted', data: 'never-render-this' }],
          content: 'Answer'
        })
      ).toEqual({
        thinkingDeltas: [],
        answerDeltas: ['Answer'],
        hasThinkingSignal: true,
        reasoningDetails: [{ type: 'reasoning.encrypted', data: 'never-render-this' }]
      })
    })

    it('uses structured Mistral thinking after encrypted-only details when no string alias exists', () => {
      expect(
        extractOpenAIThinking({
          reasoning_details: [{ type: 'reasoning.encrypted', data: 'never-render-this' }],
          content: [
            { type: 'thinking', thinking: [{ type: 'text', text: 'Structured disclosed reasoning' }] },
            { type: 'text', text: 'Answer' }
          ]
        })
      ).toEqual({
        thinkingDeltas: ['Structured disclosed reasoning'],
        answerDeltas: ['Answer'],
        hasThinkingSignal: true,
        reasoningDetails: [{ type: 'reasoning.encrypted', data: 'never-render-this' }],
        structuredContent: [
          { type: 'thinking', thinking: [{ type: 'text', text: 'Structured disclosed reasoning' }] },
          { type: 'text', text: 'Answer' }
        ]
      })
    })

    it('rejects a mixed invalid reasoning_details array and falls through to the first string alias', () => {
      expect(
        extractOpenAIThinking({
          reasoning_details: [{ type: 'reasoning.text', text: 'must not partially accept' }, null],
          reasoning: 'Validated alias'
        })
      ).toEqual({
        thinkingDeltas: ['Validated alias'],
        answerDeltas: [],
        hasThinkingSignal: true
      })
    })

    it('falls through empty textual details to an alias while retaining the detail signal and replay', () => {
      expect(
        extractOpenAIThinking({
          reasoning_details: [
            { type: 'reasoning.text', text: '' },
            { type: 'reasoning.summary', summary: '' }
          ],
          analysis: 'Fallback analysis'
        })
      ).toEqual({
        thinkingDeltas: ['Fallback analysis'],
        answerDeltas: [],
        hasThinkingSignal: true,
        reasoningDetails: [
          { type: 'reasoning.text', text: '' },
          { type: 'reasoning.summary', summary: '' }
        ]
      })
    })

    it('extracts mixed Mistral thinking and answer entries while preserving their order within each list', () => {
      const content = [
        {
          type: 'thinking',
          thinking: [
            { type: 'text', text: 'First ' },
            { type: 'ignored', text: 42 },
            { type: 'text', text: 'second.' }
          ]
        },
        { type: 'text', text: 'Answer one.' },
        { type: 'thinking', thinking: [{ type: 'text', text: ' Third.' }] },
        { type: 'text', text: ' Answer two.' },
        { type: 'unknown', text: 'ignored' }
      ]

      const result = extractOpenAIThinking({ content })

      expect(result).toEqual({
        thinkingDeltas: ['First second. Third.'],
        answerDeltas: ['Answer one.', ' Answer two.'],
        hasThinkingSignal: true,
        structuredContent: content
      })
      expect(result.structuredContent).not.toBe(content)
    })

    it('ignores structured Mistral thinking when a string alias wins but still emits structured answers', () => {
      expect(
        extractOpenAIThinking({
          reasoning: 'Parsed reasoning',
          content: [
            { type: 'thinking', thinking: [{ type: 'text', text: 'Duplicate structured reasoning' }] },
            { type: 'text', text: 'Answer' }
          ]
        })
      ).toEqual({
        thinkingDeltas: ['Parsed reasoning'],
        answerDeltas: ['Answer'],
        hasThinkingSignal: true,
        structuredContent: [
          { type: 'thinking', thinking: [{ type: 'text', text: 'Duplicate structured reasoning' }] },
          { type: 'text', text: 'Answer' }
        ]
      })
    })

    it('passes plain string content through as answer content without interpreting tags', () => {
      expect(extractOpenAIThinking({ content: '<think>parsed centrally</think>\nAnswer' })).toEqual({
        thinkingDeltas: [],
        answerDeltas: ['<think>parsed centrally</think>\nAnswer'],
        hasThinkingSignal: false
      })
    })

    it.each([null, undefined, 'string', 1, true, [], { content: null }, { content: 42 }])(
      'returns an empty extraction for unsupported input: %p',
      value => {
        expect(extractOpenAIThinking(value)).toEqual({
          thinkingDeltas: [],
          answerDeltas: [],
          hasThinkingSignal: false
        })
      }
    )

    it('ignores malformed reasoning detail text while preserving valid replay records', () => {
      expect(
        extractOpenAIThinking({
          reasoning_details: [
            { type: 'reasoning.text', text: 42 },
            { type: 'reasoning.summary', summary: null },
            { type: 'unknown', text: 'not disclosed by contract' }
          ]
        })
      ).toEqual({
        thinkingDeltas: [],
        answerDeltas: [],
        hasThinkingSignal: false,
        reasoningDetails: [
          { type: 'reasoning.text', text: 42 },
          { type: 'reasoning.summary', summary: null },
          { type: 'unknown', text: 'not disclosed by contract' }
        ]
      })
    })
  })
})

describe('LeadingThinkingTagParser', () => {
  const supportedTags = ['think', 'thinking', 'analysis'] as const

  it.each(supportedTags)('parses <%s> across every possible opening and closing tag split', tag => {
    const opening = `<${tag}>`
    const closing = `</${tag}>`
    const expected: CollectedParseResult = {
      thinking: 'inspect inventory',
      content: 'answer',
      signalEvents: 1,
      hasThinkingSignal: true
    }

    for (let split = 0; split <= opening.length; split += 1) {
      expect(
        parseChunks([opening.slice(0, split), opening.slice(split), 'inspect inventory', closing, '\nanswer'])
      ).toEqual(expected)
    }

    for (let split = 0; split <= closing.length; split += 1) {
      expect(
        parseChunks([opening, 'inspect inventory', closing.slice(0, split), closing.slice(split), '\nanswer'])
      ).toEqual(expected)
    }
  })

  it.each(supportedTags)('is invariant when the entire <%s> response is split at every character boundary', tag => {
    const input = ` \t<${tag}>inspect inventory</${tag}>\r\nanswer`
    const expected: CollectedParseResult = {
      thinking: 'inspect inventory',
      content: 'answer',
      signalEvents: 1,
      hasThinkingSignal: true
    }

    for (let split = 0; split <= input.length; split += 1) {
      expect(parseChunks([input.slice(0, split), input.slice(split)])).toEqual(expected)
    }
    expect(parseChunks([...input])).toEqual(expected)
  })

  it.each([
    ['<ThInK>Mixed case</tHiNk>Answer', 'Mixed case'],
    ['<THINKING>Mixed case</thinking>Answer', 'Mixed case'],
    ['<AnAlYsIs>Mixed case</aNaLySiS>Answer', 'Mixed case']
  ])('matches opening and closing tags case-insensitively: %s', (input, expectedThinking) => {
    expect(parseChunks([...input])).toEqual({
      thinking: expectedThinking,
      content: 'Answer',
      signalEvents: 1,
      hasThinkingSignal: true
    })
  })

  it('discards only whitespace before a recognized leading opening tag', () => {
    expect(parseChunks([' \t\r\n', '<think>', ' verbatim thought ', '</think>', 'answer'])).toEqual({
      thinking: ' verbatim thought ',
      content: 'answer',
      signalEvents: 1,
      hasThinkingSignal: true
    })
  })

  it.each([
    [['<think>x</think>', '\n', 'answer'], 'answer'],
    [['<think>x</think>', '\r', '\nanswer'], 'answer'],
    [['<think>x</think>', '\n\nanswer'], '\nanswer'],
    [['<think>x</think>', '\r\n\r\nanswer'], '\r\nanswer'],
    [['<think>x</think>', '  answer'], '  answer'],
    [['<think>x</think>', '\tanswer'], '\tanswer'],
    [['<think>x</think>', '\ranswer'], '\ranswer']
  ])('removes at most one immediate line break and preserves all other answer whitespace: %p', (chunks, content) => {
    expect(parseChunks(chunks)).toEqual({
      thinking: 'x',
      content,
      signalEvents: 1,
      hasThinkingSignal: true
    })
  })

  it.each(supportedTags)('treats an unclosed leading <%s> block as thinking at end of stream', tag => {
    expect(parseChunks([`<${tag}>`, 'unclosed ', 'thought'])).toEqual({
      thinking: 'unclosed thought',
      content: '',
      signalEvents: 1,
      hasThinkingSignal: true
    })
  })

  it.each(['</think>answer', '</thinking>answer', '</analysis>answer'])(
    'keeps a stray closing tag literal: %s',
    input => {
      expect(parseChunks([...input])).toEqual({
        thinking: '',
        content: input,
        signalEvents: 0,
        hasThinkingSignal: false
      })
    }
  )

  it.each([
    'Answer <think>literal</think>',
    '```xml\n<thinking>literal code</thinking>\n```',
    'Visible answer\n<analysis>quoted XML</analysis>',
    '  Visible answer<think>later</think>'
  ])('keeps supported-looking tags literal after answer text begins: %s', input => {
    expect(parseChunks([...input])).toEqual({
      thinking: '',
      content: input,
      signalEvents: 0,
      hasThinkingSignal: false
    })
  })

  it.each([
    '<reasoning>not supported</reasoning>',
    'Thinking: prose label',
    '[Thinking...] prose marker',
    '# Thinking\nMarkdown heading',
    '<thinker>unknown tag</thinker>'
  ])('does not recognize unsupported tags, labels, or headings: %s', input => {
    expect(parseChunks([...input])).toEqual({
      thinking: '',
      content: input,
      signalEvents: 0,
      hasThinkingSignal: false
    })
  })

  it.each(['<', '<t', '<thi', '<think', '<thinking', '<analysis'])(
    'keeps an incomplete opening tag literal at end of stream: %s',
    input => {
      expect(parseChunks([...input])).toEqual({
        thinking: '',
        content: input,
        signalEvents: 0,
        hasThinkingSignal: false
      })
    }
  )

  it('keeps every later thought cycle literal after the first closing delimiter', () => {
    expect(parseChunks(['<think>first</think>', '<analysis>later</analysis>'])).toEqual({
      thinking: 'first',
      content: '<analysis>later</analysis>',
      signalEvents: 1,
      hasThinkingSignal: true
    })
  })

  it('emits exactly one signal for an empty recognized thought block', () => {
    expect(parseChunks(['<think>', '</think>', 'Answer'])).toEqual({
      thinking: '',
      content: 'Answer',
      signalEvents: 1,
      hasThinkingSignal: true
    })
  })

  it('ignores empty pushes without changing parsing state', () => {
    expect(parseChunks(['', '<think>', '', 'thought', '', '</think>', '', 'answer', ''])).toEqual({
      thinking: 'thought',
      content: 'answer',
      signalEvents: 1,
      hasThinkingSignal: true
    })
  })
})

describe('parseLeadingThinkingTags', () => {
  it.each([
    ['<think>thought</think>\nanswer', { thinking: 'thought', content: 'answer', hasThinkingSignal: true }],
    ['<thinking>thought</thinking>answer', { thinking: 'thought', content: 'answer', hasThinkingSignal: true }],
    ['<analysis>thought', { thinking: 'thought', content: '', hasThinkingSignal: true }],
    [
      'answer <think>literal</think>',
      { thinking: '', content: 'answer <think>literal</think>', hasThinkingSignal: false }
    ],
    ['</think>stray close', { thinking: '', content: '</think>stray close', hasThinkingSignal: false }],
    ['', { thinking: '', content: '', hasThinkingSignal: false }]
  ])('returns the complete full-text parse for %p', (input, expected) => {
    expect(parseLeadingThinkingTags(input)).toEqual(expected)
  })
})
