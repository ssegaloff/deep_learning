import { Component } from 'react'

// react only offers error boundaries as class components; without one, a render error unmounts the whole app
export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    // the caller decides how to show the error, because inside a canvas only three objects (or drei's Html) can render
    if (this.state.error) return this.props.fallback(this.state.error)
    return this.props.children
  }
}
