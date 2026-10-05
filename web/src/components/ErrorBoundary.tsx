import { Component, type ReactNode } from "react";

/** Last line of defence: a rendering bug shows a message on its screen instead of blanking the app. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="note n-stop" role="alert">
        <div>
          <b>This screen hit an error and could not be drawn.</b> {this.state.error.message}. Reload the page; if it happens again,
          note what you did and the run id.
          <div style={{ marginTop: 8 }}>
            <button className="mini" onClick={() => window.location.reload()}>Reload</button>
          </div>
        </div>
      </div>
    );
  }
}
